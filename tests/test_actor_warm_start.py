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


def test_actor_uses_rolled_warm_start(device):
    torch.set_default_dtype(torch.double)
    nx = nu = 1
    dt = 0.1
    policy = DummyPolicy(nx, nu).to(device).double()
    actor = ActorMPC(nx, nu, horizon=3, dt=dt, f_dyn=f_lin, policy_net=policy, device=str(device)).to(device)

    state = torch.zeros(nx, device=device)
    actor(state)
    prev = actor.U_prev.clone()

    captured = {}
    orig_solve_step = actor.mpc.solve_step

    def wrapped(x, U_init, *args, **kwargs):
        captured['U_init'] = U_init.detach().clone()
        return orig_solve_step(x, U_init, *args, **kwargs)

    actor.mpc.solve_step = wrapped
    actor(state)

    expected = torch.roll(prev, shifts=-1, dims=1)
    expected[:, -1] = 0.0
    assert torch.allclose(captured['U_init'], expected)

