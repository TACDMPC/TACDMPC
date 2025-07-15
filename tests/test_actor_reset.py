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
        mu, log_std = self.fc(x).chunk(2, dim=-1)
        return mu, log_std


def test_actor_reset_clears_warm_start():
    torch.set_default_dtype(torch.double)
    actor = ActorMPC(1, 1, horizon=3, dt=0.1, f_dyn=f_lin, policy_net=DummyPolicy(1, 1))
    state = torch.zeros(1)
    actor(state)
    assert actor.U_prev is not None
    actor.reset()
    assert actor.U_prev is None

