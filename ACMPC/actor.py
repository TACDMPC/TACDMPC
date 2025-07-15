import torch
from torch import nn, Tensor
from torch.distributions import Normal
from DifferentialMPC.controller import DifferentiableMPCController
from DifferentialMPC.cost import GeneralQuadCost
from DifferentialMPC import GradMethod

class ActorMPC(nn.Module):
    """Actor che combina una policy neurale con un layer MPC differenziabile.

    Le matrici di costo Q e R sono parametri ottimizzabili tramite backpropagation
    (vincolo di positività implementato via Softplus). L'actor restituisce il
    primo comando ottimale combinato con l'output della policy.
    """

    def __init__(
        self,
        nx: int,
        policy_net: nn.Module,
        mpc: DifferentiableMPCController,
        *,
        time_variant: bool = False,
        deterministic: bool = False,
    ) -> None:
        super().__init__()
        self.nx = nx
        self.policy_net = policy_net
        self.mpc = mpc
        self.deterministic = deterministic
        self.time_variant = time_variant
        H, nu = mpc.horizon, mpc.nu
        if time_variant:
            self.Q_param = nn.Parameter(torch.zeros(H, nx))
            self.R_param = nn.Parameter(torch.zeros(H, nu))
        else:
            self.Q_param = nn.Parameter(torch.zeros(nx))
            self.R_param = nn.Parameter(torch.zeros(nu))
        self.softplus = nn.Softplus()
        self.register_buffer("_U_prev", torch.zeros(1, H, nu))

    # ------------------------------------------------------------------
    def _update_cost_matrices(self) -> None:
        device, dtype = self.Q_param.device, self.Q_param.dtype
        H, nx, nu = self.mpc.horizon, self.nx, self.mpc.nu
        if self.time_variant:
            Q_diag = self.softplus(self.Q_param)
            R_diag = self.softplus(self.R_param)
            Q = torch.diag_embed(Q_diag)
            R = torch.diag_embed(R_diag)
        else:
            Q_diag = self.softplus(self.Q_param)
            R_diag = self.softplus(self.R_param)
            Q = torch.diag(Q_diag)
            R = torch.diag(R_diag)
            Q = Q.expand(H, -1, -1)
            R = R.expand(H, -1, -1)
        C = torch.zeros(H, nx + nu, nx + nu, device=device, dtype=dtype)
        C[:, :nx, :nx] = Q
        C[:, nx:, nx:] = R
        c = torch.zeros(H, nx + nu, device=device, dtype=dtype)
        C_final = C[0].clone()
        c_final = torch.zeros(nx + nu, device=device, dtype=dtype)
        self.mpc.cost_module = GeneralQuadCost(
            nx, nu, C, c, C_final, c_final, device=str(device)
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        x: Tensor,
        *,
        x_ref_full: Tensor | None = None,
        u_ref_full: Tensor | None = None,
        deterministic: bool | None = None,
        return_rollout: bool = False,
    ):
        if deterministic is None:
            deterministic = self.deterministic
        single = x.ndim == 1
        if single:
            x = x.unsqueeze(0)
        self._update_cost_matrices()
        X_star, U_star = self.mpc.forward(x, x_ref_full=x_ref_full, u_ref_full=u_ref_full)
        self._U_prev = U_star.detach()
        mpc_action = U_star[:, 0]
        mu, log_std_raw = self.policy_net(x)
        log_std = log_std_raw.clamp(min=-20.0, max=2.0)
        dist = Normal(mu + mpc_action, log_std.exp())
        if deterministic or not self.training:
            action = mu + mpc_action
        else:
            action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        if single:
            action = action.squeeze(0)
            log_prob = log_prob.squeeze(0)
        if return_rollout:
            return action, log_prob, (X_star, U_star)
        return action, log_prob
