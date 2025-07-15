import torch
import torch.nn as nn
import pytest

from ACMPC import ActorMPC, CriticTransformer, training_loop


def f_lin(x, u, dt):
    return x + dt * u


class DummyPolicy(nn.Module):
    def __init__(self, nx, nu):
        super().__init__()
        self.fc = nn.Linear(nx, 2 * nu)

    def forward(self, x):
        mu, log_std_raw = self.fc(x).chunk(2, dim=-1)
        return mu, log_std_raw


class DummyEnv:
    def __init__(self, dt=0.1, device="cpu"):
        self.dt = dt
        self.device = device
        self.state = None

    def reset(self):
        self.state = torch.zeros(1, device=self.device, dtype=torch.float32)
        return self.state

    def step(self, action):
        self.state = f_lin(self.state.unsqueeze(0), action.unsqueeze(0), self.dt).squeeze(0)
        reward = -self.state.pow(2).sum().item()
        return self.state, reward, False


def test_amp_training_does_not_break_gradients(device):
    if device.type != "cuda":
        pytest.skip("AMP test requires CUDA")

    torch.set_default_dtype(torch.float32)
    env = DummyEnv(device=device)
    nx = nu = 1
    policy = DummyPolicy(nx, nu).to(device)
    actor = ActorMPC(nx, nu, horizon=3, dt=0.1, f_dyn=f_lin, policy_net=policy, device=str(device)).to(device)
    critic = CriticTransformer(nx, nu, history_len=1, pred_horizon=1).to(device)

    training_loop.train(env, actor, critic, steps=1, use_amp=True)

    grads = [p.grad for p in policy.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
