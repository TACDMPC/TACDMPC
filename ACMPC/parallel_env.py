import torch
import numpy as np
from typing import Callable, List, Tuple, Any


class ParallelEnvManager:
    """
    A manager for running multiple instances environment in parallel.

    """

    def __init__(self, env_fn: Callable, num_envs: int, device: torch.device):
        """
        Initializes N environments.
        """
        self.envs = [env_fn() for _ in range(num_envs)]
        self.num_envs = num_envs
        self.device = device

        if hasattr(self.envs[0], 'observation_space') and hasattr(self.envs[0], 'action_space'):
            self.single_observation_space_shape = self.envs[0].observation_space.shape
            self.single_action_space_shape = self.envs[0].action_space.shape
        else:
            print("Warning: The environment does not seem to have 'observation_space' or 'action_space' attributes.")

    def reset(self) -> torch.Tensor:
        """
        Resets all environments and returns a stacked tensor of observations.
        """
        # FIX: gym.reset() returns a tuple (observation, info). We only need the observation.
        observations = [env.reset()[0] for env in self.envs]
        return torch.from_numpy(np.stack(observations)).to(self.device, dtype=torch.float32)

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[dict]]:
        """
        Esegue un passo in tutti gli ambienti con le azioni fornite.
        """
        actions_np = actions.cpu().numpy()
        next_states, rewards, terminateds, truncateds, infos = [], [], [], [], []

        for i, env in enumerate(self.envs):
            ns, r, terminated, truncated, info = env.step(actions_np[i])
            next_states.append(ns)
            rewards.append(r)
            terminateds.append(terminated)
            truncateds.append(truncated)
            infos.append(info)
        return (
            torch.from_numpy(np.stack(next_states)).to(self.device, dtype=torch.float32),
            torch.from_numpy(np.array(rewards)).to(self.device, dtype=torch.float32),
            torch.from_numpy(np.array(terminateds)).to(self.device, dtype=torch.bool),
            torch.from_numpy(np.array(truncateds)).to(self.device, dtype=torch.bool),
            infos
        )

    def close(self):
        """Closes all environments."""
        [env.close() for env in self.envs]
