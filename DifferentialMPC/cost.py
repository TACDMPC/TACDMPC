# cost.py (Versione Definitiva e Robusta)

from __future__ import annotations
from typing import Optional, Tuple
from torch import Tensor
import torch


class GeneralQuadCost(torch.nn.Module):
    def __init__(self, nx: int, nu: int, C: Tensor, c: Tensor, C_final: Tensor, c_final: Tensor, **kwargs):
        super().__init__()
        self.device = torch.device(kwargs.get("device", "cpu"))
        self.nx, self.nu = nx, nu
        self.n_tau = nx + nu
        self.slew_lambda = float(kwargs.get("slew_lambda", 0.0))

        # Attributi per i costi (sovrascritti dall'attore)
        self.C, self.c = C.to(self.device), c.to(self.device)
        self.C_final, self.c_final = C_final.to(self.device), c_final.to(self.device)

        # Gestione dei riferimenti
        T_ref = C.shape[0] if C.ndim == 3 else C.shape[1]
        x_ref_default = torch.zeros(1, T_ref + 1, nx, device=self.device)
        u_ref_default = torch.zeros(1, T_ref, nu, device=self.device)
        self.register_buffer("x_ref", kwargs.get("x_ref", x_ref_default))
        self.register_buffer("u_ref", kwargs.get("u_ref", u_ref_default))
        self.set_reference(self.x_ref, self.u_ref)

    #@torch.no_grad()
    def set_reference(self, x_ref: Optional[Tensor] = None, u_ref: Optional[Tensor] = None):
        if x_ref is not None:
            if x_ref.ndim == 2: x_ref = x_ref.unsqueeze(0)
            self.x_ref = x_ref.to(self.device)
        if u_ref is not None:
            if u_ref.ndim == 2: u_ref = u_ref.unsqueeze(0)
            self.u_ref = u_ref.to(self.device)

    def _prepare_costs(self, B: int):
        """Helper per rendere i costi compatibili con il batch size dell'input."""
        C, c = self.C, self.c
        C_final, c_final = self.C_final, self.c_final

        # Se C è unbatchato (T, D, D), aggiungi una dimensione batch
        if C.ndim == 3:
            C = C.unsqueeze(0)
            c = c.unsqueeze(0)
        if C_final.ndim == 2:
            C_final = C_final.unsqueeze(0)
            c_final = c_final.unsqueeze(0)

        # Se il batch size del costo è 1 e l'input è > 1, espandi
        if C.shape[0] == 1 and B > 1:
            C = C.expand(B, -1, -1, -1)
            c = c.expand(B, -1, -1)
        if C_final.shape[0] == 1 and B > 1:
            C_final = C_final.expand(B, -1, -1)
            c_final = c_final.expand(B, -1)

        return C, c, C_final, c_final

    def objective(self, X, U, *, x_ref_override=None, u_ref_override=None):
        was_unbatched = X.ndim == 2
        if was_unbatched: X, U = X.unsqueeze(0), U.unsqueeze(0)
        B, T = X.shape[0], U.shape[1]

        C_run, c_run, C_final, c_final = self._prepare_costs(B)
        assert C_run.shape[0] == B, f"Internal Error: Batch size of C ({C_run.shape[0]}) doesn't match input X ({B})"

        x_ref = x_ref_override if x_ref_override is not None else self.x_ref
        u_ref = u_ref_override if u_ref_override is not None else self.u_ref

        dx, du = X[:, :-1] - x_ref[:, :T], U - u_ref
        tau = torch.cat((dx, du), dim=-1)
        H_sym = 0.5 * (C_run + C_run.transpose(-2, -1))
        cost_run = 0.5 * torch.einsum("bti,btij,btj->b", tau, H_sym, tau) + torch.einsum("bti,bti->b", c_run, tau)
        if self.slew_lambda > 0.0:
            ddu = torch.diff(du, dim=1, prepend=torch.zeros_like(du[:, :1]))
            cost_run += 0.5 * self.slew_lambda * (ddu ** 2).sum(dim=[1, 2])

        dxN = X[:, -1] - x_ref[:, -1]
        tauN = torch.cat([dxN, torch.zeros(B, self.nu, device=self.device, dtype=X.dtype)], dim=-1)
        Cf_sym = 0.5 * (C_final + C_final.transpose(-2, -1))
        cost_final = 0.5 * torch.einsum("bi,bij,bj->b", tauN, Cf_sym, tauN) + torch.einsum("bi,bi->b", c_final, tauN)

        total_cost = cost_run + cost_final
        return total_cost[0] if was_unbatched else total_cost

    def quadraticize(self, X, U, *, x_ref_override=None, u_ref_override=None):
        B, T = X.shape[0], U.shape[1]

        C_run, c_run, C_final, c_final = self._prepare_costs(B)
        assert C_run.shape[0] == B, f"Internal Error: Batch size of C ({C_run.shape[0]}) doesn't match input X ({B})"

        x_ref = x_ref_override if x_ref_override is not None else self.x_ref
        u_ref = u_ref_override if u_ref_override is not None else self.u_ref

        dx, du = X[:, :-1] - x_ref[:, :T], U - u_ref
        tau = torch.cat((dx, du), dim=-1)
        H_sym = 0.5 * (C_run + C_run.transpose(-2, -1))
        l_tau = torch.einsum("btij,btj->bti", H_sym, tau) + c_run
        H_tau = H_sym
        if self.slew_lambda > 0.0:
            ddu = torch.diff(du, dim=1, prepend=torch.zeros_like(du[:, :1]))
            l_tau[..., self.nx:] += 2.0 * self.slew_lambda * ddu
            H_slew = 2.0 * self.slew_lambda * torch.eye(self.nu, device=self.device, dtype=X.dtype)
            H_tau = H_sym.clone()
            H_tau[..., self.nx:, self.nx:] += H_slew

        dxN = X[:, -1] - x_ref[:, -1]
        tauN = torch.cat([dxN, torch.zeros(B, self.nu, device=self.device, dtype=X.dtype)], dim=-1)
        Cf_sym = 0.5 * (C_final + C_final.transpose(-2, -1))
        lN = torch.einsum("bij,bj->bi", Cf_sym, tauN) + c_final

        l_x, l_u = l_tau[..., :self.nx], l_tau[..., self.nx:]
        l_xx, l_uu = H_tau[..., :self.nx, :self.nx], H_tau[..., self.nx:, self.nx:]
        l_xu = H_tau[..., :self.nx, self.nx:]
        l_xN, l_xxN = lN[..., :self.nx], Cf_sym[..., :self.nx, :self.nx]

        return l_x, l_u, l_xx, l_xu, l_uu, l_xN, l_xxN