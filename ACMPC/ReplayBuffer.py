import torch
import numpy as np
from collections import deque
import random


class SequenceReplayBuffer:

    def __init__(self, buffer_size: int, history_len: int, nx: int, nu: int, device: str):
        self.buffer = deque(maxlen=buffer_size)
        self.history_len = history_len
        self.nx = nx
        self.nu = nu
        self.device = device

    def store(self, state, action, reward, next_state, done):
        state = state.flatten().astype(np.float32)
        action = action.flatten().astype(np.float32)
        next_state = next_state.flatten().astype(np.float32)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple:
        valid_indices = range(self.history_len - 1, len(self.buffer))
        batch_indices = random.sample(valid_indices, batch_size)

        history_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        for idx in batch_indices:
            # Estrai la sequenza di `history_len`
            sequence = [self.buffer[j] for j in range(idx - self.history_len + 1, idx + 1)]

            # input per il critico)
            history_states = np.array([s[0] for s in sequence])
            history_actions = np.array([s[1] for s in sequence])
            history = np.concatenate([history_states, history_actions], axis=1)
            history_batch.append(history)

            # Estrai i dati dalla transizione corrente (l'ultima della sequenza)
            current_s, current_a, current_r, next_s, current_d = sequence[-1]
            action_batch.append(current_a)
            reward_batch.append([current_r])
            next_state_batch.append(next_s)
            done_batch.append([current_d])
        return (
            torch.tensor(np.array(history_batch), dtype=torch.float64, device=self.device),
            torch.tensor(np.array(action_batch), dtype=torch.float64, device=self.device),
            torch.tensor(np.array(reward_batch), dtype=torch.float64, device=self.device),
            torch.tensor(np.array(next_state_batch), dtype=torch.float64, device=self.device),
            torch.tensor(np.array(done_batch), dtype=torch.float64, device=self.device)
        )

    def __len__(self):
        return len(self.buffer)

