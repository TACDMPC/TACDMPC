import torch
from ACMPC import ActorMPC
import importlib
mod = importlib.import_module("examples.05_double_integrator_ac_vs_baseline")


def test_train_ac_smoke():
    torch.set_default_dtype(torch.double)
    env = mod.DoubleIntegratorEnv()
    actor, rewards, q_grads = mod.train_ac(env, steps=1)
    assert isinstance(actor, ActorMPC)
    assert len(rewards) == 1
    assert len(q_grads) == 1
