import torch
from torch import nn
from DifferentialMPC.controller import DifferentiableMPCController
from DifferentialMPC.cost import GeneralQuadCost
from DifferentialMPC import GradMethod
from ACMPC  import ActorMPC       # ← adatta al tuo path

# --------------------------------------------------------------------------- #
# Dinamica lineare 1-D                                                        #
# --------------------------------------------------------------------------- #
def f_lin(x, u, dt):
    return x + dt * u                      # nx = nu = 1


def make_mpc(device):
    nx = nu = 1
    H  = 20
    dt = 0.1
    cost = GeneralQuadCost(
        nx, nu,
        C=torch.zeros(H, nx + nu, nx + nu, device=device, dtype=torch.double),
        c=torch.zeros(H, nx + nu,              device=device, dtype=torch.double),
        C_final=torch.zeros(nx + nu, nx + nu,  device=device, dtype=torch.double),
        c_final=torch.zeros(nx + nu,           device=device, dtype=torch.double),
        device=str(device),
    )
    return DifferentiableMPCController(
        f_dyn=f_lin,
        total_time=H * dt,
        step_size=dt,
        horizon=H,
        cost_module=cost,
        grad_method=GradMethod.AUTO_DIFF,
        detach_unconverged=False,          # ← non staccare nemmeno se non converge
        device=str(device),
    )


# --------------------------------------------------------------------------- #
class DummyPolicy(nn.Module):
    def __init__(self, nx, nu):
        super().__init__()
        self.fc = nn.Linear(nx, 2 * nu)

    def forward(self, x):
        mu, log_std_raw = self.fc(x).chunk(2, dim=-1)
        return mu, log_std_raw


# --------------------------------------------------------------------------- #
def test_actor_grad_flow_warm():
    torch.set_default_dtype(torch.double)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nx = nu = 1
    policy = DummyPolicy(nx, nu).to(device).double()
    mpc = make_mpc(device)

    actor = ActorMPC(
        nx=nx,
        policy_net=policy,
        mpc=mpc,
        deterministic=False,
    ).to(device).double()

    # stato iniziale batch-1
    x = torch.randn(nx, device=device, requires_grad=True)

    # ── Primo passaggio: serve solo a riempire U_prev per warm-start ──
    _ , _ = actor(x)

    # ── Secondo passaggio (con warm-start) ──
    action, _ = actor(x)

    # Loss fittizia
    loss = (action ** 2).sum()
    loss.backward()

    # ------- Asserts ---------------------------------------------------------
    assert x.grad is not None and torch.isfinite(x.grad).all()

    grads = [p.grad for p in policy.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)

    # controllo ottimo deve portare grad_fn
    X_opt, U_opt = mpc.forward(x.unsqueeze(0))
    assert U_opt.requires_grad, "u_mpc è ancora staccato dal grafo!"
