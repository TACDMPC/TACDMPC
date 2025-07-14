
# ---------- Parametri base (identici) -----------------------------------
dt, horizon, x_dim, u_dim = 0.05, 20, 4, 1
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- 1. Modello Koopman (già creato prima, riuso) ----------------
# se non esiste ancora in RAM, riesegui la cella di BlockKoopmanNet prima
koop_net = BlockKoopmanNet(
    dt=dt, x_dim=x_dim, u_dim=u_dim,
    z_dim=8, h_dim=128, aux_dim=64,
).to(device)
koop_net.eval()             # per sicurezza (niente dropout, ecc.)

# ---------- 2. Costi (come prima, con lunghezza horizon) ----------------
n_tau = x_dim + u_dim
Qx  = 10.0 * torch.eye(x_dim, device=device)
Ru  =  1.0 * torch.eye(u_dim, device=device)
C_running = torch.block_diag(Qx, Ru).repeat(horizon, 1, 1)
c_running = torch.zeros(horizon, n_tau, device=device)
C_final   = 20.0 * torch.block_diag(torch.eye(x_dim, device=device),
                                    torch.zeros(u_dim, u_dim, device=device))
c_final   = torch.zeros(n_tau, device=device)

cost_mod = GeneralQuadCost(
    nx=x_dim, nu=u_dim,
    C=C_running, c=c_running,
    C_final=C_final, c_final=c_final,
    device=device,
)

# ---------- 3. Adattatori ------------------------------------------------
def _ensure_batch(t: torch.Tensor) -> torch.Tensor:
    return t.unsqueeze(0) if t.ndim == 1 else t        # (nx,) -> (1,nx)

def f_dyn(x: torch.Tensor, u: torch.Tensor, dt_local: float) -> torch.Tensor:
    xb, ub = _ensure_batch(x), _ensure_batch(u)
    out = koop_net.dynamics(xb, ub, dt_local)
    return out.squeeze(0) if x.ndim == 1 else out

def f_dyn_jac(x: torch.Tensor, u: torch.Tensor, dt_local: float):
    xb, ub = _ensure_batch(x), _ensure_batch(u)
    A, B = koop_net.jacobians(xb, ub)
    if x.ndim == 1:   # rimuovi dim batch
        return A.squeeze(0), B.squeeze(0)
    return A, B

# ---------- 4. Controller -----------------------------------------------
controller = DifferentiableMPCController(
    f_dyn       = f_dyn,
    total_time  = horizon * dt,
    step_size   = dt,
    horizon     = horizon,
    cost_module = cost_mod,
    u_min       = torch.tensor([-1.0], device=device),
    u_max       = torch.tensor([ 1.0], device=device),
    device      = device,
    grad_method = GradMethod.ANALYTIC,
    f_dyn_jac   = f_dyn_jac,
    verbose     = 0,
)

# ---------- 5. Sanity-check ---------------------------------------------
x0     = torch.randn(x_dim, device=device)
U_init = torch.zeros(horizon, u_dim, device=device)

with torch.no_grad():
    X_traj, U_traj = controller.solve_step(x0, U_init, max_iters=5)

print("✓ MPC solve_step OK – X:", X_traj.shape, " U:", U_traj.shape)
print("  u range:", (U_traj.min().item(), U_traj.max().item()))


# ---- batch di stati iniziali -------------------------------------------
B         = 4
x0_batch  = torch.randn(B, x_dim, device=device)       # (B, nx)

# aumenta peso stato, diminuisci peso controllo
Qx = 100.0 * torch.eye(x_dim, device=device)
Ru = 0.01 * torch.eye(u_dim, device=device)
C_running[:] = torch.block_diag(Qx, Ru)        # broadcast su T

cost_mod.C.data.copy_(C_running)               # aggiorna i pesi

# risolvi con più passi di iLQR
with torch.no_grad():
    Xs_traj, Us_traj = controller.forward(x0_batch)        # usa costi aggiornati

print("u range dopo tuning:", Us_traj.min().item(), Us_traj.max().item())
