from __future__ import annotations
import torch
from torch import nn
from torch.distributions import Normal
from DifferentialMPC.controller import DifferentiableMPCController


class ActorMPC(nn.Module):
    """Policy network with differentiable MPC layer and learnable Q/R."""

    def __init__(
        self,
        nx: int,
        policy_net: nn.Module,
        mpc: DifferentiableMPCController,
        *,
        deterministic: bool = False,
    ) -> None:
        super().__init__()
        self.nx = nx
        self.nu = mpc.nu
        self.policy_net = policy_net
        self.mpc = mpc
        self.deterministic = deterministic
        H = mpc.horizon
        # learnable diagonals for Q and R (time variant)
        self.q_raw = nn.Parameter(torch.zeros(H, nx))
        self.r_raw = nn.Parameter(torch.zeros(H, self.nu))
        self.softplus = nn.Softplus()

    def _update_cost(self) -> None:
        H = self.mpc.horizon
        device = self.q_raw.device
        dtype = self.q_raw.dtype
        q_diag = self.softplus(self.q_raw)
        r_diag = self.softplus(self.r_raw)
        C = torch.zeros(H, self.nx + self.nu, self.nx + self.nu, device=device, dtype=dtype)
        C[:, : self.nx, : self.nx] = torch.diag_embed(q_diag)
        C[:, self.nx :, self.nx :] = torch.diag_embed(r_diag)
        self.mpc.cost_module.C = C
        self.mpc.cost_module.C_final = C[-1]

    def forward(
        self,
        x: torch.Tensor,
        U_init: torch.Tensor | None = None,
        *,
        deterministic: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        single = x.ndim == 1
        if single:
            x = x.unsqueeze(0)
        B = x.shape[0]
        if U_init is None:
            U_init = torch.zeros(B, self.mpc.horizon, self.nu, device=x.device, dtype=x.dtype)
        elif U_init.ndim == 2:
            U_init = U_init.unsqueeze(0).expand(B, -1, -1)
        self._update_cost()
        mu, log_std_raw = self.policy_net(x)
        log_std = torch.clamp(log_std_raw, -20.0, 2.0)
        std = log_std.exp()
        u_mpc, _ = self.mpc.solve_step(x, U_init)
        det = self.deterministic if deterministic is None else deterministic
        dist = Normal(mu + u_mpc, std)
        action = mu + u_mpc if det or not self.training else dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        if single:
            action = action.squeeze(0)
            log_prob = log_prob.squeeze(0)
        return action, log_prob
