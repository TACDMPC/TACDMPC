"""Quick demo of the new ActorMPC and CriticTransformer."""

import torch
from ACMPC import ActorMPC, CriticTransformer
from DifferentialMPC import DifferentiableMPCController
from config import MPCConfig, parse_mpc_config


def dummy_dyn(x, u, dt):
    return x + dt * u


class DummyEnv:
    def __init__(self, dt: float = 0.1):
        self.state = torch.zeros(1)
        self.dt = dt

    def reset(self):
        self.state = torch.zeros(1)
        return self.state

    def step(self, action):
        self.state = dummy_dyn(self.state, action, self.dt)
        reward = -self.state.pow(2).sum().item()
        return self.state, reward, False


def main(cfg: MPCConfig):
    device = torch.device("cpu")

    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(1, 2)

        def forward(self, x):
            mu, log_std = self.fc(x).chunk(2, dim=-1)
            return mu, log_std

    policy = Policy()
    actor = ActorMPC(
        1,
        1,
        horizon=cfg.horizon,
        dt=cfg.dt,
        f_dyn=dummy_dyn,
        policy_net=policy,
        device="cpu",
    )
    critic = CriticTransformer(1, 1, history_len=2, pred_horizon=2)
    env = DummyEnv(dt=cfg.dt)
    state = env.reset()
    action, _ = actor(state)
    print("Action sample", action)
    q = critic(
        state.unsqueeze(0),
        action.unsqueeze(0),
        torch.zeros(1, 2, 2),
        torch.zeros(1, 2, 2),
    )
    print("Critic output", q)


if __name__ == "__main__":
    cfg = parse_mpc_config()
    main(cfg)
