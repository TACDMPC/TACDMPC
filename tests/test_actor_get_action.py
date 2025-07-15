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


def test_get_action_shape_and_reproducibility(device):
    torch.set_default_dtype(torch.double)
    nx = nu = 1
    dt = 0.1
    policy = DummyPolicy(nx, nu).to(device).double()
    actor = ActorMPC(
        nx, nu, horizon=3, dt=dt, f_dyn=f_lin, policy_net=policy, device=str(device)
    ).to(device)
    actor.train()

    x_single = torch.randn(nx, device=device)
    torch.manual_seed(0)
    a1, lp1 = actor.get_action(x_single)
    torch.manual_seed(0)
    a2, lp2 = actor.get_action(x_single)
    assert a1.shape == (nu,)
    assert lp1.shape == ()
    assert torch.allclose(a1, a2)
    assert torch.allclose(lp1, lp2)

    x_batch = torch.randn(4, nx, device=device)
    torch.manual_seed(1)
    ab1, lpb1 = actor.get_action(x_batch)
    torch.manual_seed(1)
    ab2, lpb2 = actor.get_action(x_batch)
    assert ab1.shape == (4, nu)
    assert lpb1.shape == (4,)
    assert torch.allclose(ab1, ab2)
    assert torch.allclose(lpb1, lpb2)
