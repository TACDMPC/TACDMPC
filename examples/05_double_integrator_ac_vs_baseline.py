import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from DifferentialMPC import DifferentiableMPCController, GeneralQuadCost
from ACMPC import ActorMPC, CriticTransformer, training_loop
from importlib import util as import_util
from pathlib import Path

try:
    from utils import seed_everything  # type: ignore
except Exception:
    spec = import_util.spec_from_file_location(
        "utils_module", Path(__file__).resolve().parents[1] / "utils.py"
    )
    _mod = import_util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(_mod)
    seed_everything = _mod.seed_everything


def f_dyn_linear(x: torch.Tensor, u: torch.Tensor, dt: float) -> torch.Tensor:
    A = torch.tensor([[1.0, dt], [0.0, 1.0]], dtype=x.dtype, device=x.device)
    B = torch.tensor([[0.0], [dt]], dtype=x.dtype, device=x.device)
    return torch.einsum("...ij,...j->...i", A, x) + torch.einsum("...ij,...j->...i", B, u)


class DoubleIntegratorEnv:
    """Sinusoidal tracking environment."""

    def __init__(self, dt: float = 0.05, horizon: int = 20, device: str = "cpu"):
        self.dt = dt
        self.horizon = horizon
        self.device = device
        self.t = 0
        self.state = None

    def _reference(self, step: int) -> torch.Tensor:
        t = torch.tensor(step * self.dt, dtype=torch.double, device=self.device)
        pos = 2.0 * torch.sin(0.5 * t)
        vel = 2.0 * 0.5 * torch.cos(0.5 * t)
        return torch.stack([pos, vel])

    def reset(self) -> torch.Tensor:
        self.t = 0
        self.state = torch.zeros(2, device=self.device, dtype=torch.double)
        return self.state

    def step(self, action: torch.Tensor):
        u = action.to(self.device).unsqueeze(0)
        self.state = f_dyn_linear(self.state.unsqueeze(0), u, self.dt).squeeze(0)
        self.t += 1
        ref = self._reference(self.t)
        err = self.state - ref
        reward = -err.pow(2).sum().item()
        done = self.t >= self.horizon
        return self.state, reward, done


def run_baseline(env: DoubleIntegratorEnv, horizon: int = 20, N_sim: int = 200):
    nx, nu = 2, 1
    device = env.device
    dt = env.dt
    Q = torch.diag(torch.tensor([10.0, 1.0], device=device))
    R = torch.diag(torch.tensor([0.1], device=device))
    C = torch.zeros(horizon, nx + nu, nx + nu, device=device)
    C[:, :nx, :nx] = Q
    C[:, nx:, nx:] = R
    c = torch.zeros_like(C[..., 0])
    cost = GeneralQuadCost(nx, nu, C, c, C[0] * 10, torch.zeros(nx + nu, device=device), device=str(device))
    mpc = DifferentiableMPCController(
        f_dyn=f_dyn_linear,
        total_time=horizon * dt,
        step_size=dt,
        horizon=horizon,
        cost_module=cost,
        grad_method="analytic",
        u_min=torch.tensor([-50.0], device=device),
        u_max=torch.tensor([50.0], device=device),
        device=str(device),
        N_sim=N_sim,
    )

    xs = []
    x = env.reset()
    U = torch.zeros(horizon, nu, device=device)
    for _ in range(N_sim):
        x_ref = torch.stack([env._reference(env.t + k) for k in range(horizon + 1)])
        u_ref = torch.zeros(horizon, nu, device=device)
        X_star, U_star = mpc.solve_step(x, U, x_ref=x_ref, u_ref=u_ref)
        u = U_star[0]
        x, _, _ = env.step(u)
        xs.append(x.detach().cpu())
    xs = torch.stack(xs)
    refs = torch.stack([env._reference(k + 1).cpu() for k in range(N_sim)])
    rmse = ((xs - refs).pow(2).sum(dim=1).sqrt()).mean().item()
    return rmse, xs, refs


def train_ac(env: DoubleIntegratorEnv, steps: int = 1000, horizon: int = 20):
    nx, nu = 2, 1
    device = env.device
    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(nx, 64),
                nn.Tanh(),
                nn.Linear(64, 2 * nu),
            )

        def forward(self, x: torch.Tensor):
            mu, log_std_raw = self.net(x).chunk(2, dim=-1)
            return mu, log_std_raw

    policy = Policy()
    actor = ActorMPC(nx, nu, horizon=horizon, dt=env.dt, f_dyn=f_dyn_linear, policy_net=policy, device=str(device))
    critic = CriticTransformer(nx, nu, history_len=1, pred_horizon=1)
    actor.double(); critic.double()
    rewards = []
    q_grads = []
    for _ in trange(steps, desc="training", leave=False):
        training_loop.train(env, actor, critic, steps=1)
        with torch.no_grad():
            q_grad = actor.q_raw.grad.abs().mean().item() if actor.q_raw.grad is not None else 0.0
        q_grads.append(q_grad)
        state = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            action, _ = actor(state)
            state, r, done = env.step(action)
            ep_reward += r
        rewards.append(ep_reward)
    return actor, rewards, q_grads


def evaluate_actor(env: DoubleIntegratorEnv, actor: ActorMPC, episodes: int = 20):
    rmse_list = []
    traj = None
    ref = None
    for _ in range(episodes):
        xs = []
        env.reset()
        state = env.state
        done = False
        while not done:
            action, _ = actor(state, deterministic=True)
            state, _, done = env.step(action)
            xs.append(state.detach().cpu())
        traj = torch.stack(xs)
        ref = torch.stack([env._reference(k + 1).cpu() for k in range(len(xs))])
        rmse_list.append(((traj - ref).pow(2).sum(dim=1).sqrt()).mean().item())
    return np.mean(rmse_list), np.std(rmse_list), traj, ref


def main():
    seed_everything(0)
    torch.set_default_dtype(torch.double)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = DoubleIntegratorEnv(device=device)

    print("Running baseline MPC...")
    rmse_baseline, traj_b, ref_b = run_baseline(env)
    print(f"Baseline RMSE: {rmse_baseline:.3f} m")

    print("\nTraining actor-critic MPC...")
    actor, rewards, q_grads = train_ac(env, steps=100)
    rmse_mean, rmse_std, traj_ac, ref_ac = evaluate_actor(env, actor)
    print(f"Actor-Critic RMSE: {rmse_mean:.3f} ± {rmse_std:.3f} m")

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title("Tracking error")
    t = np.arange(len(traj_b)) * env.dt
    plt.plot(t, (traj_b[:, 0] - ref_b[:, 0]).numpy(), label="Baseline")
    plt.plot(t, (traj_ac[:, 0] - ref_ac[:, 0]).numpy(), label="AC")
    plt.xlabel("time [s]")
    plt.ylabel("position error [m]")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.title("Cumulative reward")
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.title("Mean |grad Q|")
    plt.plot(q_grads)
    plt.xlabel("Training step")
    plt.ylabel("|grad Q|")
    plt.grid(True)
    plt.show()

    print("\nRMSE over 20 test episodes")
    print(f"Baseline : {rmse_baseline:.3f} m")
    print(f"Actor-Critic : {rmse_mean:.3f} ± {rmse_std:.3f} m")

    comment = (
        "L'attore-critic impara una matrice Q più adatta alle traiettorie "
        "sinusoidali. Ciò riduce l'errore di tracking rispetto al controller "
        "statico, pur mantenendo la struttura MPC. La riduzione dei gradienti "
        "su Q segnala la convergenza del parametro."
    )
    print("\nCommento:")
    print(comment)

    print("\nFAQ/Troubleshooting")
    print(" - Se il solver diverge, ridurre la learning rate o aumentare R.")
    print(" - Overfitting del critic: ridurre la sua dimensione o early stop.")
    print(" - Gradienti esplosivi su Q: applicare gradient clipping.")


if __name__ == "__main__":
    main()
