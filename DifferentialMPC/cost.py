# DifferentialMPC/cost.py
from __future__ import annotations
# --- MODIFICA QUI ---
from typing import Optional, Tuple
from torch import Tensor # Importa Tensor da torch invece che da typing
# --------------------
import torch

__all__ = ["GeneralQuadCost"]

class GeneralQuadCost(torch.nn.Module):

    def __init__(
            self,
            nx: int,
            nu: int,
            C: Tensor,
            c: Tensor,
            C_final: Tensor,
            c_final: Tensor,
            *,  # keyword-only args
            slew_lambda: float = 0.0,
            learnable_cost: bool = False,
            device: str = "cpu",
            x_ref: Optional[Tensor] = None,
            u_ref: Optional[Tensor] = None,
    ) -> None:
        super().__init__()
        # Parametri base
        self.device = torch.device(device)
        self.nx, self.nu = nx, nu
        self.n_tau = nx + nu
        self.T = C.shape[0]

        # Buffers per C, c, C_final, c_final
        def _make_buffer(name: str, tensor: Tensor) -> None:
            self.register_buffer(name, tensor.to(self.device))

        _make_buffer("C", C)
        _make_buffer("c", c)
        _make_buffer("C_final", C_final)
        _make_buffer("c_final", c_final)

        self.slew_lambda = float(slew_lambda)

        # --- batch‐first default references: sempre (1, T+1, nx) e (1, T, nu)
        # x_ref
        if x_ref is None:
            x_ref = torch.zeros(self.T + 1, nx, device=self.device)
        x_ref = x_ref.to(self.device)
        if x_ref.ndim == 2:
            # da (T+1, nx) a (1, T+1, nx)
            x_ref = x_ref.unsqueeze(0)
        assert x_ref.ndim == 3 and x_ref.shape[1:] == (self.T + 1, nx), \
            f"x_ref deve essere (T+1, nx) o (B, T+1, nx), trovato {tuple(x_ref.shape)}"

        # u_ref
        if u_ref is None:
            u_ref = torch.zeros(self.T, nu, device=self.device)
        u_ref = u_ref.to(self.device)
        if u_ref.ndim == 2:
            # da (T, nu) a (1, T, nu)
            u_ref = u_ref.unsqueeze(0)
        assert u_ref.ndim == 3 and u_ref.shape[1:] == (self.T, nu), \
            f"u_ref deve essere (T, nu) o (B, T, nu), trovato {tuple(u_ref.shape)}"

        # Registrazione dei buffer batched
        self.register_buffer("x_ref", x_ref)
        self.register_buffer("u_ref", u_ref)

    # -----------------------------------------------------------------
    @torch.no_grad()  # DECORATOR
    def set_reference(
            self,
            x_ref: Optional[Tensor] = None,
            u_ref: Optional[Tensor] = None,
    ) -> None:
        """
        Aggiorna i riferimenti di stato e controllo.
        Accetta:
          - x_ref di shape (T+1, nx) o (B, T+1, nx)
          - u_ref di shape (T,   nu) o (B, T,   nu)
        e li normalizza internamente in shape (B, T+1, nx) e (B, T, nu).
        """
        # x_ref
        if x_ref is not None:
            x_ref = x_ref.to(self.device)
            if x_ref.ndim == 2:
                # da (T+1, nx) a (1, T+1, nx)
                x_ref = x_ref.unsqueeze(0)
            assert x_ref.ndim == 3 and x_ref.shape[1:] == (self.T + 1, self.nx), \
                f"x_ref deve essere (T+1, nx) o (B, T+1, nx), trovato {tuple(x_ref.shape)}"
            # assegna al buffer
            self.x_ref = x_ref

        # u_ref
        if u_ref is not None:
            u_ref = u_ref.to(self.device)
            if u_ref.ndim == 2:
                # da (T, nu) a (1, T, nu)
                u_ref = u_ref.unsqueeze(0)
            assert u_ref.ndim == 3 and u_ref.shape[1:] == (self.T, self.nu), \
                f"u_ref deve essere (T, nu) o (B, T, nu), trovato {tuple(u_ref.shape)}"
            # assegna al buffer
            self.u_ref = u_ref

    # -----------------------------------------------------------------
    def objective(
            self,
            X: Tensor,
            U: Tensor,
            *,  # Forza gli argomenti seguenti ad essere passati per nome (keyword)
            x_ref_override: Optional[Tensor] = None,
            u_ref_override: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Costo totale per batch.
        X : (B, T+1, nx)  oppure (T+1, nx)
        U : (B, T,   nu)  oppure (T,   nu)
        Return
        ------
        cost : (B,) se batch, oppure scalar se unbatched
        """
        # --- ricordo se era unbatched ---
        was_unbatched = (X.ndim == 2 and U.ndim == 2)

        # ----- normalizza batch dim ----------------------------------------
        if X.ndim == 2:
            X = X.unsqueeze(0)
        if U.ndim == 2:
            U = U.unsqueeze(0)

        B, T, nx = X.shape[0], U.shape[1], self.nx
        nu = self.nu
        device, dtype = X.device, X.dtype

        # ----- helper per broadcast temporale di C e c ----------------------
        def _bcast_time(tensor: Tensor, tgt_T: int) -> Tensor:
            tensor = tensor.to(device=device, dtype=dtype)
            if tensor.shape[0] == tgt_T:
                return tensor
            assert tensor.shape[0] == 1, \
                f"Incompatibile: prima dim di cost ({tensor.shape[0]}) vs orizzonte {tgt_T}"
            return tensor.expand(tgt_T, *tensor.shape[1:])

        # ----- broadcast delle matrici di costo -----------------------------
        C_run = _bcast_time(self.C, T)  # (T, nτ, nτ)
        c_run = _bcast_time(self.c, T)  # (T, nτ)

        # ----- MODIFICA: broadcast dei riferimenti batch‐wise ----------------
        # Usa gli override se sono forniti, altrimenti usa quelli memorizzati in `self`.
        # Questo permette a vmap di passare la slice di riferimento corretta.
        x_ref = x_ref_override if x_ref_override is not None else self.x_ref
        u_ref = u_ref_override if u_ref_override is not None else self.u_ref

        # Normalizza la dimensione batch del riferimento ricevuto
        if x_ref.ndim == 2:
            x_ref = x_ref.unsqueeze(0)
        if u_ref.ndim == 2:
            u_ref = u_ref.unsqueeze(0)

        # La logica di controllo ora funziona, perché quando chiamata da vmap,
        # sia `B` che la dimensione batch del riferimento saranno 1.
        x_ref = x_ref.to(device=device, dtype=dtype)
        if x_ref.shape[0] == 1:
            x_ref_b = x_ref.expand(B, -1, -1)
        elif x_ref.shape[0] == B:
            x_ref_b = x_ref
        else:
            raise ValueError(f"Batch di x_ref ({x_ref.shape[0]}) diverso da 1 o da B={B}")

        u_ref = u_ref.to(device=device, dtype=dtype)
        if u_ref.shape[0] == 1:
            u_ref_b = u_ref.expand(B, -1, -1)
        elif u_ref.shape[0] == B:
            u_ref_b = u_ref
        else:
            raise ValueError(f"Batch di u_ref ({u_ref.shape[0]}) diverso da 1 o da B={B}")
        # --- FINE MODIFICA ---

        # ----- errori di tracking -------------------------------------------
        dx = X[:, :-1, :] - x_ref_b[:, :-1, :]  # (B, T, nx)
        du = U - u_ref_b  # (B, T, nu)
        tau = torch.cat((dx, du), dim=-1)  # (B, T, nx+nu)

        # ----- costo di corsa -----------------------------------------------
        H_sym = 0.5 * (C_run + C_run.transpose(-2, -1))  # (T, nτ, nτ)
        H_sym_b = H_sym.unsqueeze(0).expand(B, -1, -1, -1)  # (B, T, nτ, nτ)
        cost_run = 0.5 * torch.einsum("bti,btij,btj->b", tau, H_sym_b, tau) \
                   + torch.einsum("ti,bti->b", c_run, tau)

        # ----- slew‐rate (opzionale) ---------------------------------------
        if self.slew_lambda > 0.0:
            ddu = torch.diff(du, dim=1, prepend=torch.zeros_like(du[:, :1]))
            cost_run = cost_run + 0.5 * self.slew_lambda * (ddu ** 2).sum((-1, -2))

        # ----- costo terminale ----------------------------------------------
        dxN = X[:, -1, :] - x_ref_b[:, -1, :]  # (B, nx)
        tauN = torch.cat([
            dxN,
            torch.zeros(B, nu, device=device, dtype=dtype)
        ], dim=-1)  # (B, nx+nu)
        Cf_sym = 0.5 * (self.C_final + self.C_final.transpose(-2, -1))  # (nτ, nτ)
        cost_final = 0.5 * torch.einsum("bi,ij,bj->b", tauN, Cf_sym, tauN) \
                     + torch.einsum("i,bi->b", self.c_final.to(device, dtype), tauN)

        cost = cost_run + cost_final

        # --- se input unbatched, restituisco scalar invece di [1] ---
        if was_unbatched:
            return cost[0]
        return cost

    def quadraticize(
            self,
            X: Tensor,
            U: Tensor,
            *,  # L'asterisco forza gli argomenti seguenti ad essere passati per nome (keyword)
            x_ref_override: Optional[Tensor] = None,
            u_ref_override: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Restituisce gradienti/Hessiane.
        Ora accetta i riferimenti corretti direttamente come argomenti.
        """
        T_inp, nx = X.shape[0] - 1, self.nx
        nu = self.nu
        ntau = nx + nu
        device, dtype = X.device, X.dtype

        # Usa i riferimenti passati come argomenti override se esistono.
        # Altrimenti, usa quelli memorizzati in 'self'
        x_ref2 = x_ref_override if x_ref_override is not None else self.x_ref
        u_ref2 = u_ref_override if u_ref_override is not None else self.u_ref

        # Se la chiamata arriva da vmap, x_ref2 e u_ref2 saranno già delle singole
        # traiettorie (unbatched). Questa logica gestisce anche chiamate dirette.
        if x_ref2.ndim == 3 and x_ref2.shape[0] == 1:
            x_ref2 = x_ref2.squeeze(0)
        if u_ref2.ndim == 3 and u_ref2.shape[0] == 1:
            u_ref2 = u_ref2.squeeze(0)

        assert x_ref2.ndim == 2, f"x_ref2 dovrebbe essere unbatched, ma ha shape {x_ref2.shape}"
        assert u_ref2.ndim == 2, f"u_ref2 dovrebbe essere unbatched, ma ha shape {u_ref2.shape}"

        # helper per broadcast di C e c lungo il tempo
        def _broadcast(mat: Tensor, tgt: int) -> Tensor:
            mat = mat.to(device=device, dtype=dtype)
            if mat.shape[0] == tgt:
                return mat
            assert mat.shape[0] == 1, \
                f"Incompatibile: first dim {mat.shape[0]} vs horizon {tgt}"
            return mat.expand(tgt, *mat.shape[1:])

        C_run = _broadcast(self.C, T_inp)
        c_run = _broadcast(self.c, T_inp)

        # errori di tracking
        dx = X[:-1] - x_ref2[:T_inp]
        du = U - u_ref2
        tau = torch.cat((dx, du), dim=-1)

        # Hessiane simmetriche
        H_sym = 0.5 * (C_run + C_run.transpose(-2, -1))

        # gradienti di corsa
        l_tau = c_run + torch.einsum("tij,tj->ti", H_sym, tau)

        # slew‐rate opzionale
        if self.slew_lambda > 0.0:
            ddu = torch.diff(du, dim=0, prepend=torch.zeros_like(du[:1]))
            l_tau[:, nx:] += 2.0 * self.slew_lambda * ddu
            H_slew = 2.0 * self.slew_lambda * torch.eye(nu, device=device, dtype=dtype)
            H_tau = H_sym.clone()
            H_tau[:, nx:, nx:] += H_slew
        else:
            H_tau = H_sym

        # costo terminale
        dxN = X[-1] - x_ref2[T_inp]
        tauN = torch.cat((dxN, torch.zeros(nu, device=device, dtype=dtype)), dim=0)
        Cf_sym = 0.5 * (self.C_final + self.C_final.transpose(-2, -1))
        lN = self.c_final.to(device, dtype) + tauN @ Cf_sym.T

        return l_tau, H_tau, lN, Cf_sym
