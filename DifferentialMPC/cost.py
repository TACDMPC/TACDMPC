from __future__ import annotations
from typing import Optional, Tuple
from torch import Tensor
import torch


class GeneralQuadCost(torch.nn.Module):
    def __init__(self, nx: int, nu: int, C: Tensor, c: Tensor, C_final: Tensor, c_final: Tensor, **kwargs):
        super().__init__()
        self.device = torch.device(kwargs.get("device", "cpu"))
        self.nx, self.nu = nx, nu
        self.C, self.c = C.to(self.device), c.to(self.device)
        self.C_final, self.c_final = C_final.to(self.device), c_final.to(self.device)
        T_ref = C.shape[0] if C.ndim == 3 else C.shape[1]
        x_ref_default = torch.zeros(1, T_ref + 1, nx, device=self.device)
        u_ref_default = torch.zeros(1, T_ref, nu, device=self.device)
        self.register_buffer("x_ref", kwargs.get("x_ref", x_ref_default))
        self.register_buffer("u_ref", kwargs.get("u_ref", u_ref_default))
        self.set_reference(self.x_ref, self.u_ref)

    def set_reference(self, x_ref: Optional[Tensor] = None, u_ref: Optional[Tensor] = None):
        if x_ref is not None:
            if x_ref.ndim == 2: x_ref = x_ref.unsqueeze(0)
            self.x_ref = x_ref.to(self.device)
        if u_ref is not None:
            if u_ref.ndim == 2: u_ref = u_ref.unsqueeze(0)
            self.u_ref = u_ref.to(self.device)

    def _prepare_costs(self, B: int):
        C, c, C_final, c_final = self.C, self.c, self.C_final, self.c_final
        if C.ndim == 3: C, c = C.unsqueeze(0), c.unsqueeze(0)
        if C_final.ndim == 2: C_final, c_final = C_final.unsqueeze(0), c_final.unsqueeze(0)
        if C.shape[0] == 1 and B > 1: C, c = C.expand(B, -1, -1, -1), c.expand(B, -1, -1)
        if C_final.shape[0] == 1 and B > 1: C_final, c_final = C_final.expand(B, -1, -1), c_final.expand(B, -1)
        return C, c, C_final, c_final

    def objective(self, X, U, *, x_ref_override=None, u_ref_override=None):
        _x_ref = x_ref_override if x_ref_override is not None else self.x_ref
        _u_ref = u_ref_override if u_ref_override is not None else self.u_ref
        was_unbatched = X.ndim == 2

        if was_unbatched:
            X, U = X.unsqueeze(0), U.unsqueeze(0)
            if _x_ref.ndim == 2: _x_ref = _x_ref.unsqueeze(0)
            if _u_ref.ndim == 2: _u_ref = _u_ref.unsqueeze(0)

        B, T = X.shape[0], U.shape[1]
        if _x_ref.shape[0] == 1 and B > 1: _x_ref = _x_ref.expand(B, -1, -1)
        if _u_ref.shape[0] == 1 and B > 1: _u_ref = _u_ref.expand(B, -1, -1)

        C_run, c_run, C_final, c_final = self._prepare_costs(B)

        dx, du = X[:, :T] - _x_ref[:, :T], U - _u_ref
        tau = torch.cat((dx, du), dim=-1)
        cost_run = 0.5 * torch.einsum("bti,btij,btj->b", tau, C_run, tau) + torch.einsum("bti,bti->b", c_run, tau)

        dxN = X[:, -1] - _x_ref[:, -1]
        tauN = torch.cat([dxN, torch.zeros(B, self.nu, device=X.device, dtype=X.dtype)], dim=-1)
        cost_final = 0.5 * torch.einsum("bi,bij,bj->b", tauN, C_final, tauN) + torch.einsum("bi,bi->b", c_final, tauN)

        total_cost = cost_run + cost_final
        return total_cost[0] if was_unbatched else total_cost

    def quadraticize(self, X, U, *, x_ref_override=None, u_ref_override=None):
        _x_ref = x_ref_override if x_ref_override is not None else self.x_ref
        _u_ref = u_ref_override if u_ref_override is not None else self.u_ref
        was_unbatched = X.ndim == 2

        if was_unbatched:
            X, U = X.unsqueeze(0), U.unsqueeze(0)
            if _x_ref.ndim == 2: _x_ref = _x_ref.unsqueeze(0)
            if _u_ref.ndim == 2: _u_ref = _u_ref.unsqueeze(0)

        B, T = X.shape[0], U.shape[1]
        if _x_ref.shape[0] == 1 and B > 1: _x_ref = _x_ref.expand(B, -1, -1)
        if _u_ref.shape[0] == 1 and B > 1: _u_ref = _u_ref.expand(B, -1, -1)

        C_run, c_run, C_final, c_final = self._prepare_costs(B)

        dx, du = X[:, :T] - _x_ref[:, :T], U - _u_ref
        tau = torch.cat((dx, du), dim=-1)

        l_tau = torch.einsum("btij,btj->bti", C_run, tau) + c_run
        H_tau = C_run

        dxN = X[:, -1] - _x_ref[:, -1]
        tauN = torch.cat([dxN, torch.zeros(B, self.nu, device=X.device, dtype=X.dtype)], dim=-1)
        lN = torch.einsum("bij,bj->bi", C_final, tauN) + c_final
        HN = C_final

        l_x, l_u = l_tau[..., :self.nx], l_tau[..., self.nx:]
        l_xx, l_uu = H_tau[..., :self.nx, :self.nx], H_tau[..., self.nx:, self.nx:]
        l_xu = H_tau[..., :self.nx, self.nx:]
        l_xN, l_xxN = lN[..., :self.nx], HN[..., :self.nx, :self.nx]

        if was_unbatched:
            return l_x.squeeze(0), l_u.squeeze(0), l_xx.squeeze(0), l_xu.squeeze(0), l_uu.squeeze(0), l_xN.squeeze(
                0), l_xxN.squeeze(0)

        return l_x, l_u, l_xx, l_xu, l_uu, l_xN, l_xxN