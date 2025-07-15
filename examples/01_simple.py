"""Quick example showing instantiation of actor and critic."""
from DifferentialMPC.controller import DifferentiableMPCController
from DifferentialMPC.cost import GeneralQuadCost
import torch
from ACMPC.actor import ActorMPC
from ACMPC.critic_transformer import CriticTransformer


def main() -> None:
    nx = 1
    nu = 1
    H = 3
    dt = 0.1

    def f_dyn(x, u, dt):
        return x + dt * u

    cost = GeneralQuadCost(
        nx,
        nu,
        C=torch.eye(nx + nu).repeat(H, 1, 1),
        c=torch.zeros(H, nx + nu),
        C_final=torch.eye(nx + nu),
        c_final=torch.zeros(nx + nu),
    )
    mpc = DifferentiableMPCController(
        f_dyn=f_dyn,
        total_time=H * dt,
        step_size=dt,
        horizon=H,
        cost_module=cost,
        device="cpu",
    )

    class SimplePolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(nx, 2 * nu)

        def forward(self, x):
            out = self.fc(x)
            return out.chunk(2, dim=-1)

    policy = SimplePolicy()
    actor = ActorMPC(nx=nx, policy_net=policy, mpc=mpc)
    critic = CriticTransformer(nx=nx, nu=nu, history_len=4)
    x = torch.randn(nx)
    action, logp = actor(x)
    print("action", action)
    h = torch.zeros(1, 4, nx + nu)
    q = critic(h)
    print("q", q)


if __name__ == "__main__":
    main()
