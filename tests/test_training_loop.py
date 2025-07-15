import torch
import torch.nn as nn

from ACMPC import ActorMPC, CriticTransformer
from ACMPC import training_loop


def f_lin(x, u, dt):
    return x + dt * u


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 2)

    def forward(self, x):
        mu, log_std = self.fc(x).chunk(2, dim=-1)
        return mu, log_std


class TinyEnv:
    def __init__(self):
        self.state = torch.zeros(1, dtype=torch.double)
        self.dt = 0.1

    def reset(self):
        self.state = torch.zeros(1, dtype=torch.double)
        return self.state.clone()

    def step(self, action):
        action = action.detach()
        new_state = f_lin(self.state, action, self.dt)
        reward = -new_state.pow(2).sum().item()
        self.state = new_state.detach().clone()
        return new_state.clone(), reward, False


def compute_loss(env, actor, critic):
    states, actions, rewards = training_loop.rollout(env, actor, horizon=5)
    returns = rewards.flip(0).cumsum(0).flip(0)
    hist = torch.zeros(1, critic.history_len, actor.nx + actor.nu)
    pred = torch.zeros(1, critic.pred_horizon, actor.nx + actor.nu)
    value = critic(states[-1].unsqueeze(0), actions[-1].unsqueeze(0), hist, pred)
    adv = returns.sum() - value
    return (-adv + adv.pow(2)).item()


def _train_step(env, actor, critic, opt_a, opt_c):
    states, actions, rewards = training_loop.rollout(env, actor, horizon=5)
    returns = rewards.flip(0).cumsum(0).flip(0)
    hist = torch.zeros(1, critic.history_len, actor.nx + actor.nu)
    pred = torch.zeros(1, critic.pred_horizon, actor.nx + actor.nu)
    value = critic(states[-1].unsqueeze(0), actions[-1].unsqueeze(0), hist, pred)
    adv = returns.sum() - value

    actor_loss = -adv
    critic_loss = adv.pow(2)

    opt_a.zero_grad()
    actor_loss.backward(retain_graph=True)
    opt_c.zero_grad()
    critic_loss.backward()
    opt_a.step()
    opt_c.step()

    return (actor_loss + critic_loss).item()


def test_training_loop_improves_loss():
    torch.set_default_dtype(torch.double)
    torch.manual_seed(0)
    policy = Policy()
    actor = ActorMPC(
        1, 1, horizon=2, dt=0.1, f_dyn=f_lin, policy_net=policy, device="cpu"
    )
    critic = CriticTransformer(1, 1, history_len=1, pred_horizon=1)
    actor.double()
    critic.double()

    opt_a = torch.optim.Adam(actor.parameters())
    opt_c = torch.optim.Adam(critic.parameters())

    env = TinyEnv()
    initial_loss = compute_loss(TinyEnv(), actor, critic)
    for _ in range(20):
        _train_step(env, actor, critic, opt_a, opt_c)
    final_loss = compute_loss(TinyEnv(), actor, critic)

    assert final_loss < initial_loss
