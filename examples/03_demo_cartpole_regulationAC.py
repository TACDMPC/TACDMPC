import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
from dataclasses import dataclass
import gym
import os
from contextlib import redirect_stdout
from utils import seed_everything
from DifferentialMPC import DifferentiableMPCController, GeneralQuadCost, GradMethod

LOG_STD_MAX = 2
LOG_STD_MIN = -20
ACTION_SCALE = 20.0  # Scala l'output di tanh [-1, 1] alla forza reale (es. [-20, 20])


# =============================================================================
# 1. DEFINIZIONE DELL'AMBIENTE CARTPOLE (dal tuo script)
# =============================================================================
@dataclass(frozen=True)
class CartPoleParams:
    m_c: float;
    m_p: float;
    l: float;
    g: float

    @classmethod
    def from_gym(cls):
        with open(os.devnull, 'w') as f, redirect_stdout(f):
            env = gym.make("CartPole-v1")
        return cls(m_c=float(env.unwrapped.masscart), m_p=float(env.unwrapped.masspole),
                   l=float(env.unwrapped.length), g=float(env.unwrapped.gravity))


def f_cartpole_dyn(x: torch.Tensor, u: torch.Tensor, dt: float, p: CartPoleParams) -> torch.Tensor:
    """Dinamica non lineare del Cart-Pole, gestisce il batching."""
    pos, vel, theta, omega = torch.unbind(x, dim=-1)
    force = u.squeeze(-1)
    sin_t, cos_t = torch.sin(theta), torch.cos(theta)
    total_mass, m_p_l = p.m_c + p.m_p, p.m_p * p.l
    temp = (force + m_p_l * omega ** 2 * sin_t) / total_mass
    theta_dd = (p.g * sin_t - cos_t * temp) / (p.l * (4.0 / 3.0 - p.m_p * cos_t ** 2 / total_mass))
    vel_dd = temp - m_p_l * theta_dd * cos_t / total_mass
    return torch.stack((pos + vel * dt, vel + vel_dd * dt, theta + omega * dt, omega + theta_dd * dt), dim=-1)


class CartPoleEnv:
    """Wrapper per la dinamica del CartPole per simulazioni RL."""

    def __init__(self, dt=0.05, device="cpu"):
        self.params = CartPoleParams.from_gym()
        self.dt = dt
        self.device = device
        self.state = None
        self.dyn_func = lambda x, u, dt: f_cartpole_dyn(x, u, dt, self.params)

    def reset(self):
        # Stato iniziale: carrello al centro, palo leggermente inclinato
        self.state = torch.tensor([0.0, 0.0, 0.2, 0.0], device=self.device, dtype=torch.float64)
        return self.state

    def step(self, action: torch.Tensor):
        if self.state is None: raise RuntimeError("Chiamare reset() prima di step().")

        # La dinamica richiede input batch, quindi aggiungiamo una dimensione
        state_batch = self.state.unsqueeze(0)
        action_batch = action.to(self.device).unsqueeze(0)

        self.state = self.dyn_func(state_batch, action_batch, self.dt).squeeze(0)

        pos, _, theta, _ = torch.unbind(self.state)
        # Ricompensa per stare vicino al centro e con il palo dritto
        reward = torch.exp(-pos.abs()) + torch.exp(-theta.abs()) - 0.001 * (action ** 2).sum()

        done = bool(pos.abs() > 2.4 or theta.abs() > 0.6)
        return self.state, reward.item(), done


# =============================================================================
# 2. DEFINIZIONE DELL'ATTORE E DELLA POLICY (versioni finali)
# =============================================================================
class ActorNet(nn.Module):
    def __init__(self, nx: int, nu: int, h_dim: int = 256):
        super().__init__()
        self.nx, self.nu = nx, nu
        self.base_network = nn.Sequential(nn.Linear(nx, h_dim), nn.ReLU(), nn.Linear(h_dim, h_dim), nn.ReLU())
        self.mean_head = nn.Linear(h_dim, nu)
        self.log_std_head = nn.Linear(h_dim, nu)
        self.cost_head = nn.Sequential(nn.Linear(h_dim, nx + nu), nn.Softplus())

    def forward(self, state: torch.Tensor):
        x = self.base_network(state)
        mean = self.mean_head(x)
        log_std = torch.clamp(self.log_std_head(x), LOG_STD_MIN, LOG_STD_MAX)
        cost_diagonals = self.cost_head(x) + 1e-6
        return mean, log_std, cost_diagonals[..., :self.nx], cost_diagonals[..., self.nx:]


class ActorMPC(nn.Module):
    def __init__(self, nx: int, nu: int, horizon: int, dt: float, f_dyn, device="cpu"):
        super().__init__()
        self.nx, self.nu, self.horizon, self.device = nx, nu, horizon, device
        self.actor_net = ActorNet(nx, nu).to(device)

        dummy_C = torch.eye(nx + nu, device=device).repeat(horizon, 1, 1)
        cost_module = GeneralQuadCost(nx=nx, nu=nu, C=dummy_C, c=torch.zeros_like(dummy_C[..., 0]),
                                      C_final=dummy_C[0], c_final=torch.zeros_like(dummy_C[0, ..., 0]), device=device)

        self.mpc = DifferentiableMPCController(
            f_dyn=f_dyn, cost_module=cost_module, horizon=horizon, step_size=dt,
            grad_method=GradMethod.FINITE_DIFF, device=device, total_time=horizon * dt,
            u_min=torch.tensor([-ACTION_SCALE]), u_max=torch.tensor([ACTION_SCALE])
        )

    def get_action(self, state: torch.Tensor, U_init: torch.Tensor, deterministic=False):
        state = state.to(self.device)
        state_b = state.unsqueeze(0) if state.ndim == 1 else state

        mean, log_std, q_diag, r_diag = self.actor_net(state_b)

        Q_mat = torch.diag_embed(q_diag.squeeze(0))
        R_mat = torch.diag_embed(r_diag.squeeze(0))
        C_step = torch.block_diag(Q_mat, R_mat)
        self.mpc.cost_module.C = C_step.unsqueeze(0).repeat(self.horizon, 1, 1)
        self.mpc.cost_module.C_final = C_step * 10.0

        with torch.no_grad():  # Il solve dell'MPC è solo una guida, non vogliamo gradienti da qui
            mpc_action, _ = self.mpc.solve_step(state, U_init.to(self.device))

        final_mean = mean.squeeze(0) + mpc_action.detach()
        std = log_std.exp().squeeze(0)
        policy_dist = Normal(final_mean, std)

        action_sample = final_mean if deterministic else policy_dist.rsample()
        log_prob = policy_dist.log_prob(action_sample).sum(axis=-1)

        action_tanh = torch.tanh(action_sample)
        log_prob -= torch.log(1 - action_tanh.pow(2) + 1e-6).sum(axis=-1)

        # Scala l'azione all'intervallo fisico del sistema
        final_action = action_tanh * ACTION_SCALE

        return final_action, log_prob


# --- Sanity Check ---
if __name__ == '__main__':
    seed_everything(0)
    torch.set_default_dtype(torch.float64)
    env = CartPoleEnv(device="cpu")
    state = env.reset()

    actor = ActorMPC(nx=4, nu=1, horizon=15, dt=env.dt, f_dyn=env.dyn_func, device="cpu")

    U_init = torch.zeros(15, 1)
    action, log_p = actor.get_action(state, U_init)

    print("Sanity Check Step 4.1")
    print(f"Stato iniziale: {state.numpy()}")
    print(f"Azione campionata: {action.detach().numpy()}")
    print(f"Log Probabilità: {log_p.item()}")

    assert action.shape == (1,)
    assert log_p.ndim == 0
    print("\n✅ Sanity check superato. L'ambiente e l'attore sono pronti.")