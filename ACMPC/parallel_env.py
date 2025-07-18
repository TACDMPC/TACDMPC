import torch
import numpy as np
from typing import Callable, List, Tuple, Any

class ParallelEnvManager:
    """
    Un gestore per eseguire più istanze di un ambiente in parallelo.
    """

    def __init__(self, env_fn: Callable, num_envs: int, device: torch.device):
        """
        Inizializza N ambienti.
        """
        self.envs = [env_fn() for _ in range(num_envs)]
        self.num_envs = num_envs
        self.device = device

        # Estrae le dimensioni dall'ambiente
        # Assumiamo che l'ambiente sia conforme all'API di Gym
        if hasattr(self.envs[0], 'observation_space') and hasattr(self.envs[0], 'action_space'):
            self.single_observation_space_shape = self.envs[0].observation_space.shape
            self.single_action_space_shape = self.envs[0].action_space.shape
        else:
            print("Attenzione: l'ambiente non sembra avere 'observation_space' o 'action_space'.")


    def reset(self) -> torch.Tensor:
        """
        Resetta tutti gli ambienti.
        """
        states = [env.reset() for env in self.envs]
        return torch.from_numpy(np.stack(states)).to(self.device, dtype=torch.float32)

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[dict]]:
        """
        Esegue uno step in tutti gli ambienti con le azioni fornite.
        """
        actions_np = actions.cpu().numpy()

        # Liste per raccogliere i risultati
        next_states, rewards, dones, infos = [], [], [], []

        for i, env in enumerate(self.envs):
            # L'API standard di Gym restituisce 4 valori
            ns, r, d, info = env.step(actions_np[i])

            next_states.append(ns)
            rewards.append(r)
            dones.append(d)
            infos.append(info)

        # --- MODIFICA CHIAVE: Restituisce anche la lista di 'infos' ---
        return (
            torch.from_numpy(np.stack(next_states)).to(self.device, dtype=torch.float32),
            torch.from_numpy(np.array(rewards)).to(self.device, dtype=torch.float32),
            torch.from_numpy(np.array(dones)).to(self.device, dtype=torch.bool),
            infos  # Il quarto valore restituito
        )

    def close(self):
        """Chiude tutti gli ambienti."""
        [env.close() for env in self.envs]