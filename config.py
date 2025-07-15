from __future__ import annotations

from dataclasses import dataclass
import argparse
from typing import Sequence


@dataclass
class MPCConfig:
    dt: float = 0.05
    horizon: int = 20
    N_sim: int = 150


@dataclass
class OptimizerConfig:
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4


@dataclass
class TrainingConfig:
    steps: int = 100
    rollout_horizon: int = 10
    mpc: MPCConfig = MPCConfig()
    optim: OptimizerConfig = OptimizerConfig()


def parse_mpc_config(args: Sequence[str] | None = None) -> MPCConfig:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dt", type=float, default=MPCConfig.dt)
    parser.add_argument("--horizon", type=int, default=MPCConfig.horizon)
    parser.add_argument("--n-sim", type=int, default=MPCConfig.N_sim)
    ns, _ = parser.parse_known_args(args)
    return MPCConfig(dt=ns.dt, horizon=ns.horizon, N_sim=ns.n_sim)


def parse_training_config(args: Sequence[str] | None = None) -> TrainingConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=TrainingConfig.steps)
    parser.add_argument(
        "--rollout-horizon", type=int, default=TrainingConfig.rollout_horizon
    )
    parser.add_argument("--actor-lr", type=float, default=OptimizerConfig.actor_lr)
    parser.add_argument("--critic-lr", type=float, default=OptimizerConfig.critic_lr)
    parser.add_argument("--dt", type=float, default=MPCConfig.dt)
    parser.add_argument("--horizon", type=int, default=MPCConfig.horizon)
    parser.add_argument("--n-sim", type=int, default=MPCConfig.N_sim)
    ns = parser.parse_args(args)
    mpc = MPCConfig(dt=ns.dt, horizon=ns.horizon, N_sim=ns.n_sim)
    optim = OptimizerConfig(actor_lr=ns.actor_lr, critic_lr=ns.critic_lr)
    return TrainingConfig(
        steps=ns.steps, rollout_horizon=ns.rollout_horizon, mpc=mpc, optim=optim
    )
