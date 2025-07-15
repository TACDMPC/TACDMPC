import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal
from DifferentialMPC import DifferentiableMPCController, GradMethod
from DifferentialMPC.cost import GeneralQuadCost


class ActorMPC(nn.Module):
    """Differentiable MPC actor with learnable Q and R diagonals."""

    def __init__(
        self,
        nx: int,
        nu: int,
        horizon: int,
        dt: float,
        f_dyn,
        policy_net: nn.Module,
        device: str = "cpu",
        deterministic: bool = False,
    ) -> None:
        super().__init__()
        self.nx = nx
        self.nu = nu
        self.horizon = horizon
        self.dt = dt
        self.device = torch.device(device)
        self.policy_net = policy_net
        self.deterministic = deterministic

        # learnable cost diagonals (time varying)
        self.q_raw = nn.Parameter(torch.zeros(horizon + 1, nx, device=self.device, dtype=torch.double))
        self.r_raw = nn.Parameter(torch.zeros(horizon, nu, device=self.device, dtype=torch.double))

        C, c, C_final, c_final = self._build_cost_mats()
        cost = GeneralQuadCost(
            nx=nx,
            nu=nu,
            C=C,
            c=c,
            C_final=C_final,
            c_final=c_final,
            device=device,
        )

        self.mpc = DifferentiableMPCController(
            f_dyn=f_dyn,
            total_time=horizon * dt,
            step_size=dt,
            horizon=horizon,
            cost_module=cost,
            grad_method=GradMethod.AUTO_DIFF,
            device=device,
        )
        self.U_prev: Tensor | None = None

    # ------------------------------------------------------------------
    def _softplus_diag(self, raw: Tensor) -> Tensor:
        return torch.nn.functional.softplus(raw) + 1e-6

    def _build_cost_mats(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        q = self._softplus_diag(self.q_raw)
        r = self._softplus_diag(self.r_raw)
        C = torch.zeros(
            self.horizon, self.nx + self.nu, self.nx + self.nu, device=self.device
        )
        for t in range(self.horizon):
            C[t, : self.nx, : self.nx] = torch.diag(q[t])
            C[t, self.nx :, self.nx :] = torch.diag(r[t])
        c = torch.zeros(self.horizon, self.nx + self.nu, device=self.device)
        C_final = torch.zeros(self.nx + self.nu, self.nx + self.nu, device=self.device)
        C_final[: self.nx, : self.nx] = torch.diag(q[-1])
        c_final = torch.zeros(self.nx + self.nu, device=self.device)
        return C, c, C_final, c_final

    def _update_cost(self) -> None:
        C, c, C_final, c_final = self._build_cost_mats()
        self.mpc.cost_module.C = C
        self.mpc.cost_module.c = c
        self.mpc.cost_module.C_final = C_final
        self.mpc.cost_module.c_final = c_final

    # ------------------------------------------------------------------
    def _soft_clamp_log_std(self, log_std_raw: Tensor) -> Tensor:
        log_std_min = -20.0
        log_std_max = 2.0
        log_std = log_std_min + torch.nn.functional.softplus(log_std_raw - log_std_min)
        log_std = log_std_max - torch.nn.functional.softplus(log_std_max - log_std)
        return log_std

    # ------------------------------------------------------------------
    def get_action(
        self,
        x: Tensor,
        U_init: Tensor | None = None,
        deterministic: bool | None = None,
        return_entropy: bool = False,
    ):
        single = x.ndim == 1
        if single:
            x = x.unsqueeze(0)
        self._update_cost()
        B = x.shape[0]
        mu, log_std_raw = self.policy_net(x)
        log_std = self._soft_clamp_log_std(log_std_raw)
        std = log_std.exp()
        if U_init is None:
            U_init = torch.zeros(B, self.horizon, self.nu, device=x.device, dtype=x.dtype)
        else:
            if U_init.ndim == 2:
                U_init = U_init.unsqueeze(0)
            if U_init.shape[0] != B:
                U_init = U_init.expand(B, -1, -1)
        if self.U_prev is not None and self.U_prev.shape[0] == B:
            U_init = self.U_prev
        u_mpc, _ = self.mpc.solve_step(x, U_init)
        self.U_prev = self.mpc.U_last.detach()
        det = self.deterministic if deterministic is None else deterministic
        dist = Normal(mu + u_mpc, std)
        if det or not self.training:
            action = mu + u_mpc
        else:
            action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        if single:
            action = action.squeeze(0)
            log_prob = log_prob.squeeze(0)
            entropy = entropy.squeeze(0)
        if return_entropy:
            return action, log_prob, entropy
        return action, log_prob

    # alias for nn.Module forward
    def forward(self, x: Tensor, U_init: Tensor | None = None, deterministic: bool | None = None):
        return self.get_action(x, U_init, deterministic)
