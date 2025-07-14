import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
from dataclasses import dataclass
import gym
import os
from contextlib import redirect_stdout

from DifferentialMPC import DifferentiableMPCController, GeneralQuadCost, GradMethod
from torch.distributions import Normal
from torch.cuda.amp import autocast
from torch import Tensor
LOG_STD_MAX = 2
LOG_STD_MIN = -20
ACTION_SCALE = 20.0

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


class ActorMPC(torch.nn.Module):
    """Actor neurale + suggerimento Differentiable MPC (grad-safe)."""

    def __init__(
        self,
        nx: int,
        policy_net: torch.nn.Module,
        mpc,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        deterministic: bool = False,
    ) -> None:
        super().__init__()
        self.nx = nx
        self.policy_net = policy_net
        self.mpc = mpc
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.deterministic = deterministic

    # ------------------------------------------------------------------ #
    def forward(
        self,
        x: Tensor,
        U_init: Tensor | None = None,
        deterministic: bool | None = None,
    ) -> tuple[Tensor, Tensor]:  # alias per nn.Module
        return self.get_action(x, U_init=U_init, deterministic=deterministic)

    # ------------------------------------------------------------------ #
    def _soft_clamp_log_std(self, log_std_raw: Tensor) -> Tensor:
        """Soft-clamp continuo e derivabile su [log_std_min, log_std_max]."""
        # porta il valore sopra il minimo con Softplus
        log_std = self.log_std_min + torch.nn.functional.softplus(
            log_std_raw - self.log_std_min
        )
        # se serve, porta anche sotto il massimo (seconda Softplus)
        log_std = self.log_std_max - torch.nn.functional.softplus(
            self.log_std_max - log_std
        )
        return log_std

    # ------------------------------------------------------------------ #
    def get_action(
        self,
        x: Tensor,
        U_init: Tensor | None = None,
        deterministic: bool | None = None,
    ) -> tuple[Tensor, Tensor]:
        single = x.ndim == 1
        if single:
            x = x.unsqueeze(0)                       # (1, nx)

        B = x.shape[0]

        # 1 ─ policy network (AMP-friendly)
        with autocast(enabled=torch.is_autocast_enabled()):
            mu, log_std_raw = self.policy_net(x)     # (B, nu) ×2

        # 2 ─ soft-clamp del log-σ (gradiente sempre ≠ 0)
        log_std = self._soft_clamp_log_std(log_std_raw)
        std = log_std.exp()

        # 3 ─ suggerimento MPC: primo controllo ottimo
        if U_init is None:
            U_init = torch.zeros(B, self.mpc.horizon, self.mpc.nu,
                                 device=x.device, dtype=x.dtype)
        else:
            if U_init.ndim == 2:
                U_init = U_init.unsqueeze(0)
            if U_init.shape[0] != B:
                U_init = U_init.expand(B, -1, -1)

        u_mpc, _ = self.mpc.solve_step(x, U_init)    # (B, nu)

        # 4 ─ composizione
        det = self.deterministic if deterministic is None else deterministic
        dist = Normal(mu + u_mpc, std)
        if det or not self.training:
            action = mu + u_mpc
        else:
            action = dist.rsample()                  # reparam-trick

        log_prob = dist.log_prob(action).sum(dim=-1)

        if single:
            action = action.squeeze(0)               # (nu,)
            log_prob = log_prob.squeeze(0)

        return action, log_prob
