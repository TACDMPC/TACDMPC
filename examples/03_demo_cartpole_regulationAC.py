"""Minimal demo showing ActorMPC and CriticTransformer usage."""
import torch
import torch.nn as nn
from dataclasses import dataclass
import gym
from ACMPC import ActorMPC, CriticTransformer
from DifferentialMPC.controller import DifferentiableMPCController
from DifferentialMPC.cost import GeneralQuadCost
from DifferentialMPC import GradMethod

@dataclass
class CartPoleParams:
    m_c: float
    m_p: float
    l: float
    g: float

    @classmethod
    def from_gym(cls):
        env = gym.make("CartPole-v1")
        return cls(float(env.unwrapped.masscart), float(env.unwrapped.masspole), float(env.unwrapped.length), float(env.unwrapped.gravity))


def f_cartpole(x: torch.Tensor, u: torch.Tensor, dt: float, p: CartPoleParams) -> torch.Tensor:
    pos, vel, theta, omega = torch.unbind(x, dim=-1)
    force = u.squeeze(-1)
    sin_t, cos_t = torch.sin(theta), torch.cos(theta)
    total_mass, m_p_l = p.m_c + p.m_p, p.m_p * p.l
    temp = (force + m_p_l * omega ** 2 * sin_t) / total_mass
    theta_dd = (p.g * sin_t - cos_t * temp) / (p.l * (4.0/3.0 - p.m_p * cos_t ** 2 / total_mass))
    vel_dd = temp - m_p_l * theta_dd * cos_t / total_mass
    return torch.stack((pos + vel * dt,
                        vel + vel_dd * dt,
                        theta + omega * dt,
                        omega + theta_dd * dt), dim=-1)

def main():
    torch.set_default_dtype(torch.float64)
    device = "cpu"
    params = CartPoleParams.from_gym()
    dyn = lambda x, u, dt: f_cartpole(x, u, dt, params)

    nx, nu, horizon, dt = 4, 1, 15, 0.05
    dummy_cost = GeneralQuadCost(
        nx, nu,
        C=torch.eye(nx+nu, device=device).repeat(horizon,1,1),
        c=torch.zeros(horizon, nx+nu, device=device),
        C_final=torch.eye(nx+nu, device=device),
        c_final=torch.zeros(nx+nu, device=device),
        device=device,
    )
    mpc = DifferentiableMPCController(
        f_dyn=dyn,
        total_time=horizon*dt,
        step_size=dt,
        horizon=horizon,
        cost_module=dummy_cost,
        grad_method=GradMethod.AUTO_DIFF,
        device=device,
    )

    class PolicyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(nx, 2 * nu)

        def forward(self, x):
            return self.fc(x).chunk(2, dim=-1)

    policy = PolicyNet()
    actor = ActorMPC(nx=nx, policy_net=policy, mpc=mpc)
    critic = CriticTransformer(nx, nu, history_len=5, horizon=horizon)

    state = torch.zeros(nx)
    action, _ = actor(state)
    value = critic(state.unsqueeze(0), torch.zeros(nu).unsqueeze(0),
                   torch.zeros(1,5,nx), torch.zeros(1,5,nu),
                   torch.zeros(1,horizon,nx), torch.zeros(1,horizon,nu))
    print("sample action", action.detach().cpu().tolist())
    print("critic value", value.item())

if __name__ == "__main__":
    main()
