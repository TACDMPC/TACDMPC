import torch, gymnasium as gym
from pathlib import Path
from tqdm.auto import trange

# ───────── settings ─────────
dtype, device = torch.float32, ("cuda" if torch.cuda.is_available() else "cpu")
dt, horizon   = 0.05, 20
x_dim, u_dim  = 3, 1
n_tau         = x_dim + u_dim          # 4

# ───────── helpers ─────────
to_f32 = lambda t: t.to(dtype)
bx     = lambda x: x.unsqueeze(0) if x.ndim == 1 else x          # (3,)→(1,3)
bu     = lambda u: (u.unsqueeze(-1) if u.ndim == 1 else           # (B,)→(B,1)
                    u.squeeze(-1) if u.ndim == 3 else u)         # (B,1,1)→(B,1)

# ───────── Koopman model (pre-train altrove) ─────────
koop_net = BlockKoopmanNet(dt, x_dim, u_dim, z_dim=6).to(device).eval().float()

# ───────── costi 4 × 4 (time-invariant) ─────────
C_one   = torch.diag(torch.tensor([10,10,10, .01], dtype=dtype, device=device)).unsqueeze(0)
C_final = torch.diag(torch.tensor([20,20,20, 0  ], dtype=dtype, device=device))
cost_mod = GeneralQuadCost(
    nx=x_dim, nu=u_dim,
    C=C_one,   c=torch.zeros_like(C_one[:,0]),
    C_final=C_final, c_final=torch.zeros(n_tau, dtype=dtype, device=device),
    device=device,
)
cost_mod.x_ref = torch.zeros(1, horizon+1, x_dim, dtype=dtype, device=device)
cost_mod.u_ref = torch.zeros(1, horizon,   u_dim, dtype=dtype, device=device)

# ───────── Koopman wrappers (safe names) ─────────
def pend_dyn(x, u, dt):                          # always batch inside
    return koop_net.dynamics(bx(x), bu(u), dt).squeeze(0)

def pend_jac(x, u, dt):
    A, B = koop_net.jacobians(bx(x), bu(u))
    return A.squeeze(0), B.squeeze(0)

# ───────── controller (uses pend_dyn / pend_jac) ─────────
controller = DifferentiableMPCController(
    f_dyn=pend_dyn,
    f_dyn_jac=pend_jac,                # se instabile, metti None e FINITE_DIFF
    step_size=dt,
    total_time=horizon*dt,
    horizon=horizon,
    cost_module=cost_mod,
    u_min=torch.tensor([-2.0], dtype=dtype, device=device),
    u_max=torch.tensor([ 2.0], dtype=dtype, device=device),
    device=device,
    grad_method=GradMethod.ANALYTIC,   # oppure FINITE_DIFF
    verbose=0,
)
print("✓ Controller creato (nτ = 4, dtype float32)")

# ───────── dataset generation ─────────
env         = gym.make("Pendulum-v1")
n_rollouts  = 1000
save_dir    = Path("data_mpc_expert"); save_dir.mkdir(exist_ok=True)

for ep in trange(n_rollouts, desc="roll-out"):
    obs, _  = env.reset(seed=ep)
    U_prev  = torch.zeros(1, horizon, u_dim, dtype=dtype, device=device)  # warm-start
    xs, us  = [], []

    for _ in range(horizon):
        x_t = to_f32(torch.tensor(obs, device=device)).unsqueeze(0)
        with torch.no_grad():
            u_opt, _ = controller.solve_step(x_t, U_prev)                 # (1,1)
        u_scalar = float(u_opt.squeeze())
        obs, _, term, trunc, _ = env.step([u_scalar])

        xs.append(x_t.squeeze(0).cpu())
        us.append(torch.tensor([u_scalar], dtype=dtype))

        # shift warm-start
        U_prev = torch.roll(U_prev, shifts=-1, dims=1)
        U_prev[:,-1,:] = u_opt

        if term or trunc: break

    xs.append(to_f32(torch.tensor(obs)).cpu())
    torch.save((torch.stack(xs).to(torch.float32),
                torch.stack(us).to(torch.float32)),
               save_dir / f"rollout_{ep:05d}.pt")

    if ep % 1000 == 0 and ep > 0:
        print(f" > salvati {ep} roll-out …")

print(f"✓ {n_rollouts} roll-out salvati in «{save_dir}» — pronti al pre-training")
from torch.utils.data import Dataset, DataLoader

class ExpertRolloutDS(Dataset):
    """Carica (X, U) di lunghezza variabile e restituisce triple (x, u, x′)."""
    def __init__(self, root: str):
        self.files = sorted(Path(root).glob("rollout_*.pt"))

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        X, U = torch.load(self.files[idx], map_location="cpu")  # X:(T+1,3) U:(T,1)
        x   = X[:-1]                    # (T , 3)
        x_p = X[1:]                     # (T , 3)
        u   = U                         # (T , 1)
        return x, u, x_p                # shapes (T,3) (T,1) (T,3)

def collate(batch):
    """Concatena le sequenze su dim0 (più efficiente di padding)."""
    xs, us, xps = zip(*batch)
    return torch.cat(xs), torch.cat(us), torch.cat(xps)

ds      = ExpertRolloutDS("data_mpc_expert")
loader  = DataLoader(ds, batch_size=16, shuffle=True,
                     collate_fn=collate, pin_memory=True)
print("Batch di esempio:", next(iter(loader))[0].shape)   # (~16*T, 3)
# %% Training MSE ----------------------------------------------------------
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW

net       = koop_net.train()            # riusa l’istanza già creata
opt       = AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
scaler    = GradScaler()
epochs    = 100

for ep in range(epochs):
    running = 0.0
    for x, u, x_p in loader:
        x, u, x_p = x.to(device), u.to(device), x_p.to(device)
        with autocast():                                # mixed-precision
            pred = net.dynamics(x, u, dt)               # (N,3)
            loss = F.mse_loss(pred, x_p)

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad()
        running += loss.item()

    print(f"[{ep+1:02d}/{epochs}]  loss = {running/len(loader):.5f}")

torch.save(net.state_dict(), "koopman_pretrained.pt")
print("✓ pesi salvati")
net.eval()
with torch.no_grad():
    x, u, x_p = next(iter(loader))
    x, u, x_p = x.to(device), u.to(device), x_p.to(device)
    err = F.mse_loss(net.dynamics(x, u, dt), x_p).sqrt()
print("RMSE a rollout 1-step:", err.item())
koop_net.load_state_dict(torch.load("koopman_pretrained.pt"))
koop_net.eval()