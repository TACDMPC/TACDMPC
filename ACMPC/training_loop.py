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


def _rollout_with_logprobs(env, actor: ActorMPC, horizon: int):
    """Rollout policy collecting log probabilities."""
    states = []
    actions = []
    rewards = []
    log_probs = []
    state = env.reset()
    for _ in range(horizon):
        action, log_prob = actor(state)
        next_state, reward, done = env.step(action)
        states.append(state)
        actions.append(action)
        rewards.append(torch.tensor([reward], dtype=state.dtype, device=state.device))
        log_probs.append(log_prob)
        state = next_state
        if done:
            state = env.reset()
    return (
        torch.stack(states),
        torch.stack(actions),
        torch.stack(rewards),
        torch.stack(log_probs),
    )


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
                states, actions, rewards, log_probs = _rollout_with_logprobs(
                    env, actor, horizon=10
                )

                returns = rewards.flip(0).cumsum(0).flip(0).squeeze(-1)

                pred = torch.zeros(
                    1, critic.pred_horizon, actor.nx + actor.nu, device=states.device
                )

                states_detached = states.detach()
                actions_detached = actions.detach()
                tokens = torch.cat([states_detached, actions_detached], dim=-1)
                values = []
                for t in range(states.shape[0]):
                    start = max(0, t - critic.history_len)
                    hist = torch.zeros(
                        critic.history_len,
                        actor.nx + actor.nu,
                        device=states.device,
                        dtype=states.dtype,
                    )
                    hist_slice = tokens[start:t]
                    if hist_slice.numel() > 0:
                        hist[-hist_slice.shape[0] :] = hist_slice
                    value = critic(
                        states_detached[t].unsqueeze(0),
                        actions_detached[t].unsqueeze(0),
                        hist.unsqueeze(0),
                        pred,
                    )
                    values.append(value.squeeze(0))
                values = torch.stack(values)

                advantages = returns - values

                actor_loss = -(log_probs * advantages.detach()).mean()
                critic_loss = torch.nn.functional.mse_loss(values, returns)

        actor_opt.zero_grad()
        critic_opt.zero_grad()

        if use_amp:
            scaler.scale(actor_loss).backward()
            scaler.scale(critic_loss).backward()
            scaler.step(actor_opt)
            scaler.step(critic_opt)
            scaler.update()
        else:
            actor_loss.backward()
            critic_loss.backward()
            actor_opt.step()
            critic_opt.step()
