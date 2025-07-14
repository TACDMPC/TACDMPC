import torch
import torch.nn as nn


class ActorNet(nn.Module):


    def __init__(self, nx: int, nu: int, h_dim: int = 256):
        super().__init__()
        self.nx = nx
        self.nu = nu

        # rete MLP (Multi-Layer Perceptron)
        self.network = nn.Sequential(
            nn.Linear(nx, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, nx + nu),
            nn.Softplus()
        )

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cost_diagonals = self.network(state)
        cost_diagonals = cost_diagonals + 1e-6 # noise
        q_diag = cost_diagonals[..., :self.nx]
        r_diag = cost_diagonals[..., self.nx:]
        return q_diag, r_diag


