import torch
from torch import nn
from DifferentialMPC.controller import DifferentiableMPCController
from DifferentialMPC.cost import GeneralQuadCost
from DifferentialMPC import GradMethod
from ACMPC import ActorMPC               # ← adatta al tuo path
class DummyPolicy(nn.Module):
    def __init__(self, nx: int, nu: int) -> None:
        super().__init__()
        self.fc = nn.Linear(nx, 2 * nu)             # out → [mu || log_std_raw]

    def forward(self, x: torch.Tensor):
        out = self.fc(x)
        mu, log_std_raw = out.chunk(2, dim=-1)
        return mu, log_std_raw
def f_lin(x, u, dt):
    return x + dt * u                               # nx=nu=1


def make_mpc(device):
    nx = nu = 1
    H = 3
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
        device=str(device),
    )
def test_actor_grad_flow():
    torch.set_default_dtype(torch.double)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nx = nu = 1
    policy = DummyPolicy(nx, nu).to(device).double()
    mpc = make_mpc(device)

    actor = ActorMPC(
        nx=nx,
        policy_net=policy,
        mpc=mpc,
        deterministic=False,         # test anche del sampling (reparam-trick)
    ).to(device).double()

    # stato batch-1
    x = torch.randn(nx, device=device, requires_grad=True)

    action = actor(x)                # grad through MPC + policy

    # fittizia loss quadratica sull'azione
    loss = (action ** 2).sum()
    loss.backward()

    # -------------- ASSERT GRADIENT FLOW ----------------------------------- #
    assert x.grad is not None and torch.isfinite(x.grad).all()
    grads = [p.grad for p in policy.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
