"""Minimal PPO training loop for ActorMPC and CriticTransformer."""

import torch
import torch.optim as optim
from contextlib import nullcontext
from importlib import import_module, util as import_util
from pathlib import Path
from utils.profiler import Profiler

try:
    from utils import seed_everything  # type: ignore
except Exception:  # package shadowing
    spec = import_util.spec_from_file_location(
        "utils_module", Path(__file__).resolve().parents[1] / "utils.py"
    )
    _mod = import_util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(_mod)
    seed_everything = _mod.seed_everything


from .actor import ActorMPC
from .critic_transformer import CriticTransformer


def rollout(env, actor: ActorMPC, horizon: int):
    states = []
    actions = []
    rewards = []
    state = env.reset()
    for _ in range(horizon):
        action, _ = actor(state)
        next_state, reward, done = env.step(action)
        states.append(state)
        actions.append(action)
        rewards.append(torch.tensor([reward], dtype=torch.float32, device=state.device))
        state = next_state
        if done:
            state = env.reset()
    return torch.stack(states), torch.stack(actions), torch.stack(rewards)


def train(
    env,
    actor: ActorMPC,
    critic: CriticTransformer,
    steps: int = 100,
    *,
    use_amp: bool = False,
    profile: bool = False,
    log_file: str | None = None,
    track_gpu: bool = False,
    seed: int | None = None,
):

    if seed is not None:
        seed_everything(seed)

    actor_opt = optim.Adam(actor.parameters(), lr=3e-4)
    critic_opt = optim.Adam(critic.parameters(), lr=3e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for _ in range(steps):
        with torch.cuda.amp.autocast(enabled=use_amp):
            ctx = (
                Profiler(
                    log_file=log_file,
                    track_gpu=track_gpu,
                    batches=1,
                )
                if profile
                else nullcontext()
            )
            with ctx:
                states, actions, rewards = rollout(env, actor, horizon=10)
                # simple cumulative reward
                returns = rewards.flip(0).cumsum(0).flip(0)
                critic_in_hist = torch.zeros(
                    1, critic.history_len, actor.nx + actor.nu, device=states.device
                )
                pred = torch.zeros(
                    1, critic.pred_horizon, actor.nx + actor.nu, device=states.device
                )
                values = critic(
                    states[-1].unsqueeze(0),
                    actions[-1].unsqueeze(0),
                    critic_in_hist,
                    pred,
                )
                advantages = returns.sum() - values

                actor_loss = -advantages
                critic_loss = advantages.pow(2)

        actor_opt.zero_grad()
        critic_opt.zero_grad()

        if use_amp:
            scaler.scale(actor_loss).backward(retain_graph=True)
            scaler.scale(critic_loss).backward()
            scaler.step(actor_opt)
            scaler.step(critic_opt)
            scaler.update()
        else:
            actor_loss.backward(retain_graph=True)
            critic_loss.backward()
            actor_opt.step()
            critic_opt.step()


