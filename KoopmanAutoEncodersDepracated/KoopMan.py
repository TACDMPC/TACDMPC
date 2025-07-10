# BlockKoopmanNet – versione self-contained (no torch.compile)

# %%
from __future__ import annotations
import torch
from torch import nn, Tensor
from torch.func import jacrev, vmap
from typing import Tuple

class BlockKoopmanNet(nn.Module):
    """
    Deep Koopman auto-encoder con matrici A(x), B(x) dipendenti dallo stato.
    Adattato da kcpo-icml (commit 2023-07-10) con:
        • attivazioni SiLU al posto di GELU
        • nessun torch.compile (massima compatibilità)
        • supporto batch > 1
    """
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        dt: float,
        x_dim: int,
        u_dim: int,
        z_dim: int = 8,
        h_dim: int = 128,
        aux_dim: int = 64,
        *,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        self.dt      = float(dt)
        self.x_dim   = x_dim
        self.u_dim   = u_dim
        self.z_dim   = z_dim
        self.n_block = z_dim // 2                     # coppie compl.-coniugate
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        act = nn.SiLU

        # ---------------- Encoder / Decoder ---------------------------- #
        self.encoder = nn.Sequential(
            nn.Linear(x_dim, h_dim, device=device), act(),
            nn.Linear(h_dim, h_dim, device=device), act(),
            nn.Linear(h_dim, z_dim, device=device),
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, h_dim, device=device), act(),
            nn.Linear(h_dim, h_dim, device=device), act(),
            nn.Linear(h_dim, h_dim, device=device), act(),
            nn.Linear(h_dim, x_dim, device=device),
        )

        # --------------- reti ausiliarie per A(x) e B(x) --------------- #
        self.aux_nn = nn.Sequential(
            nn.Linear(x_dim, aux_dim, device=device), act(),
            nn.Linear(aux_dim, aux_dim, device=device), act(),
            nn.Linear(aux_dim, z_dim, device=device),                # → A-params
        )
        self.bux_nn = nn.Sequential(
            nn.Linear(x_dim, aux_dim, device=device), act(),
            nn.Linear(aux_dim, aux_dim, device=device), act(),
            nn.Linear(aux_dim, z_dim * u_dim, device=device),        # → B-params
        )

    # ================================================================== #
    # ------------------------ helper interni -------------------------- #
    def _rot_scale_block(self, a: Tensor, b: Tensor) -> Tensor:
        """
        Restituisce blocco 2×2 = exp(a·dt) · R(b·dt) dove
        R(θ) = [[cosθ, −sinθ], [sinθ, cosθ]].
        `a`, `b` scalari (tensor 0-dim o 1-dim).
        """
        t = self.dt
        factor = torch.exp(a * t)
        c, s   = torch.cos(b * t), torch.sin(b * t)
        return factor * torch.stack(
            (torch.stack((c, -s)), torch.stack((s, c)))
        )

    # ================================================================== #
    # --------------------------- API base ----------------------------- #
    def encode(self, x: Tensor) -> Tensor:            # φ
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:            # φ⁻¹
        return self.decoder(z)

    # ---------------- A(x) e B(x) ------------------------------------- #
    def A(self, x: Tensor) -> Tensor:
        """
        A(x) block-diagonal per batch.
        Output: (B, z_dim, z_dim)
        """
        aux = self.aux_nn(x)                          # (B, z_dim)
        Bsz = aux.shape[0]
        mats: list[Tensor] = []

        for k in range(Bsz):                          # loop batch
            vec = aux[k]                              # (z_dim,)
            blocks = [
                self._rot_scale_block(vec[2*i], vec[2*i + 1])
                for i in range(self.n_block)
            ]
            mats.append(torch.block_diag(*blocks))

        return torch.stack(mats, dim=0)               # (B, z_dim, z_dim)

    def B(self, x: Tensor) -> Tensor:
        """B(x)  →  shape (B, z_dim, u_dim)."""
        return self.bux_nn(x).view(-1, self.z_dim, self.u_dim)

    # ---------------- dynamics & jacobians ---------------------------- #
    def latent_step(self, z: Tensor, u: Tensor, x: Tensor) -> Tensor:
        return z + self.dt * (
            torch.bmm(self.A(x), z.unsqueeze(-1)).squeeze(-1) +
            torch.bmm(self.B(x), u.unsqueeze(-1)).squeeze(-1)
        )

    def dynamics(self, x: Tensor, u: Tensor, dt: float | None = None) -> Tensor:
        """
        f(x,u) → x′  (se dt è diverso da self.dt viene usato temporaneamente).
        """
        dt = float(dt) if dt is not None else self.dt
        if dt != self.dt:
            old_dt, self.dt = self.dt, dt
        z_next = self.latent_step(self.encode(x), u, x)
        if dt != self.dt:
            self.dt = old_dt
        return self.decode(z_next)

    # ---------------- jacobiani batch-wise ---------------------------- #
    def jacobians(self, x: Tensor, u: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Restituisce tuple (A, B) con
            A ≡ ∂x′/∂x  shape (B, x_dim, x_dim)
            B ≡ ∂x′/∂u  shape (B, x_dim, u_dim)
        """
        # helper senza batch
        def f_x(x_single: Tensor, u_single: Tensor) -> Tensor:
            return self.dynamics(x_single.unsqueeze(0),
                                 u_single.unsqueeze(0)).squeeze(0)

        # ∂f/∂x
        A_batch = vmap(lambda _x, _u:
                       jacrev(lambda _x0: f_x(_x0, _u), argnums=0)(_x))(x, u)

        # ∂f/∂u
        B_batch = vmap(lambda _x, _u:
                       jacrev(lambda _u0: f_x(_x, _u0), argnums=0)(_u))(x, u)

        return A_batch, B_batch
