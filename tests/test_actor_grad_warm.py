import torch
import torch.nn as nn
from ACMPC import ActorMPC


def f_lin(x, u, dt):
    return x + dt * u


class DummyPolicy(nn.Module):
    def __init__(self, nx, nu):
        super().__init__()
        self.fc = nn.Linear(nx, 2 * nu)

    def forward(self, x):
        mu, log_std_raw = self.fc(x).chunk(2, dim=-1)
        return mu, log_std_raw


def test_actor_grad_flow_warm(device):
    torch.set_default_dtype(torch.double)
    nx = nu = 1
    dt = 0.1
    policy = DummyPolicy(nx, nu).to(device).double()
    actor = ActorMPC(nx, nu, horizon=20, dt=dt, f_dyn=f_lin, policy_net=policy, device=str(device)).to(device)

    x = torch.randn(nx, device=device, requires_grad=True)
    _ = actor(x)
    action, _ = actor(x)
    loss = (action ** 2).sum()
    loss.backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    grads = [p.grad for p in policy.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    X_opt, U_opt = actor.mpc.forward(x.unsqueeze(0))
    assert U_opt.requires_grad
