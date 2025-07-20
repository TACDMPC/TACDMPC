from __future__ import annotations
import enum
import logging
from typing import Callable, Optional, Tuple, Dict
import torch
from torch import Tensor
from torch.func import jacrev
try:
    from torch.func import vmap as _vmap
    _HAS_VMAP = True
except ImportError:
    _HAS_VMAP = False
    _vmap = None
from .utils import pnqp, jacobian_finite_diff_batched
from .cost import GeneralQuadCost
try:
    from torch.func import scan
    _HAS_SCAN = True
except ImportError:
    _HAS_SCAN = False

def _outer(a: Tensor, b: Tensor) -> Tensor:
    return a.unsqueeze(-1) * b.unsqueeze(-2)

""" vecchia implementazione i gradienti non passano
class ILQRSolve(torch.autograd.Function):

    @staticmethod
    def forward(ctx,
                x0: torch.Tensor,
                controller: 'DifferentiableMPCController',
                U_init: torch.Tensor) -> torch.Tensor:
        u_opt, _ = controller.solve_step(x0, U_init)
        # stacca se non convergente
        if (not controller.converged) and controller.detach_unconverged:
            u_opt = u_opt.detach()

        # tensori per backward
        ctx.controller = controller
        # Questo ora funzionerà perché self.H_last ecc. sono stati impostati da solve_step
        ctx.save_for_backward(
            x0,
            controller.X_last,
            controller.U_last,
            controller.H_last[0], controller.H_last[1], controller.H_last[2],
            controller.F_last[0], controller.F_last[1],
            controller.lmb_last,
            controller.tight_mask_last
        )
        return u_opt
    # ------------------------------------------------------------------
    @staticmethod
    def backward(ctx, grad_u_out: torch.Tensor):
        (
            _x0,
            _Xs, Us,                     # Us serve solo per shape di grad_U_init
            H_xx, H_uu, H_xu,
            A,  Bm,
            _lmb,
            tight_mask,
        ) = ctx.saved_tensors
        ctrl = ctx.controller

        B, T, nx, nu = A.shape[0], A.shape[1], ctrl.nx, ctrl.nu
        device, dtype = A.device, A.dtype

        grad_x0_list = []

        # ------------------------------------------------------------------ #
        # loop esplicito sul batch (niente vmap -> ok control-flow dinamico)  #
        # ------------------------------------------------------------------ #
        for i in range(B):
            # 1. Costruisci gradiente su τ_u: solo sul primo comando
            grad_tau_u = torch.cat(
                [grad_u_out[i].unsqueeze(0),
                 torch.zeros(T - 1, nu, dtype=dtype, device=device)],
                dim=0
            )

            # 2. Patch temporaneo di U_last (shape (T,nu) attesa dallo solver)
            U_last_orig = ctrl.U_last
            if U_last_orig.ndim == 3:           # (B, T, nu)
                ctrl.U_last = U_last_orig[i]    # (T, nu)

            # 3. Backward LQR singolo batch
            dX, _, _ = ctrl._zero_constrained_lqr(
                A[i], Bm[i],
                H_xx[i], H_uu[i], H_xu[i],
                torch.zeros(T + 1, nx, dtype=dtype, device=device),  # grad τ_x = 0
                grad_tau_u,
                tight_mask[i],
                delta_u=ctrl.delta_u,
            )

            grad_x0_list.append(dX[0])          # ∂ℒ/∂x0 per questo batch

            # 4. Ripristina U_last originale
            ctrl.U_last = U_last_orig

        # ------------------------------------------------------------------ #
        # Aggrega risultati e restituisci                                    #
        # ------------------------------------------------------------------ #
        grad_x0 = torch.stack(grad_x0_list, dim=0)       # (B, nx)
        grad_U_init = torch.zeros_like(Us)               # non ottimizziamo U_init

        # ordine: grad_x0, grad_controller(None), grad_U_init
        return grad_x0, None, grad_U_init
"""


class ILQRSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                x0: Tensor,
                # I parametri da apprendere sono ora input diretti
                C: Tensor, c: Tensor, C_final: Tensor, c_final: Tensor,
                # Il controller è un "worker", non un input da differenziare
                controller: 'DifferentiableMPCController',
                U_init: Tensor
                ) -> Tuple[Tensor, Tensor]:

        # 1. Assegna i costi al modulo del controller per il forward pass
        # N.B. I costi possono avere una dimensione batch
        controller.cost_module.C = C
        controller.cost_module.c = c
        controller.cost_module.C_final = C_final
        controller.cost_module.c_final = c_final

        # 2. Risolvi l'MPC per ottenere le traiettorie ottimali
        X_opt, U_opt = controller.solve_step(x0, U_init)

        # 3. Stacca i risultati dal grafo se il solver non converge
        if (not controller.converged) and controller.detach_unconverged:
            X_opt, U_opt = X_opt.detach(), U_opt.detach()

        # 4. Salva i tensori per il backward pass
        ctx.controller = controller
        # Salviamo i risultati ottimali e i parametri che hanno generato la soluzione
        ctx.save_for_backward(
            X_opt, U_opt,
            controller.H_last[0], controller.H_last[1], controller.H_last[2],  # H_xx, H_uu, H_xu
            controller.F_last[0], controller.F_last[1],  # A, Bm
            controller.tight_mask_last
        )

        # 5. L'output della funzione sono le traiettorie ottimali
        return X_opt, U_opt

    @staticmethod
    def backward(ctx, grad_X_out: Tensor, grad_U_out: Tensor):
        """
        Implementazione completa del backward pass basata sulla teoria di DIFFMPC.
        """
        # 1. Recupera i dati salvati
        X, U, H_xx, H_uu, H_xu, A, Bm, tight_mask = ctx.saved_tensors
        ctrl = ctx.controller
        B = X.shape[0]

        # =================================================================
        # =============== INIZIO CODICE DI DEBUG ==========================
        # =================================================================
        print("\n--- DEBUG: Dentro ILQRSolve.backward ---")
        print(f"Norma gradiente in ingresso (grad_U_out): {torch.linalg.norm(grad_U_out).item():.4e}")
        # =================================================================

        # 2. Combina i gradienti in ingresso in un unico tensore `r`
        r_x = grad_X_out[:, :-1, :]
        r_u = grad_U_out

        # 3. Risolvi l'LQR all'indietro
        dX_list, dU_list = [], []
        for i in range(B):
            dX_i, dU_i, _ = ctrl._zero_constrained_lqr(
                A[i], Bm[i], H_xx[i], H_uu[i], H_xu[i],
                -r_x[i], -r_u[i], tight_mask[i]
            )
            dX_list.append(dX_i)
            dU_list.append(dU_i)

        dX = torch.stack(dX_list)
        dU = torch.stack(dU_list)

        # =================================================================
        # =============== ALTRO CODICE DI DEBUG ===========================
        # =================================================================
        print(f"Norma sensitività calcolata (dU): {torch.linalg.norm(dU).item():.4e}")
        # =================================================================

        # 4. Calcola i gradienti dei parametri di costo
        dtau = torch.cat([dX[:, :-1, :], dU], dim=-1)
        tau = torch.cat([X[:, :-1, :], U], dim=-1)

        grad_C = -0.5 * (_outer(dtau, tau) + _outer(tau, dtau))
        grad_c = -dtau

        # =================================================================
        # =============== CODICE DI DEBUG FINALE ==========================
        # =================================================================
        print(f"Norma gradiente finale per C (grad_C): {torch.linalg.norm(grad_C).item():.4e}")
        print("----------------------------------------\n")
        # =================================================================

        # ... il resto della funzione rimane invariato
        grad_C_final_xx = -0.5 * (_outer(dX[:, -1], X[:, -1]) + _outer(X[:, -1], dX[:, -1]))
        grad_C_final = torch.nn.functional.pad(grad_C_final_xx, (0, ctrl.nu, 0, ctrl.nu))
        grad_c_final = torch.nn.functional.pad(-dX[:, -1], (0, ctrl.nu))
        grad_x0 = dX[:, 0, :]

        return grad_x0, grad_C, grad_c, grad_C_final, grad_c_final, None, None


# ─────────────────────────────────────────────────────────────
class GradMethod(enum.Enum):
    """Modalità di calcolo della Jacobiana (A, B) della dinamica f(x,u,dt).

    * **ANALYTIC**  : l'utente fornisce una funzione `f_dyn_jac(x,u,dt)` che
                      restituisce (A, B) in forma analitica ‑faster and better .
    * **AUTO_DIFF** : usa `torch.autograd.functional.jacobian` (default) VERY SLOW.
    * **FINITE_DIFF**: differenze finite centralizzate con passo `fd_eps` LEGACY.
    """
    ANALYTIC = "analytic"
    AUTO_DIFF = "auto_diff"
    FINITE_DIFF = "finite_diff"


class DifferentiableMPCController(torch.nn.Module):
    # -------------------------------------------------------------------
    def __init__(
            self,
            f_dyn: Callable,
            total_time: float,
            step_size: float,
            horizon: int,
            cost_module: torch.nn.Module,
            u_min: Optional[torch.Tensor] = None,
            u_max: Optional[torch.Tensor] = None,
            reg_eps: float = 1e-6,
            device: str = "cuda:0",
            N_sim: Optional[int] = None,
            grad_method: GradMethod | str = GradMethod.AUTO_DIFF,
            f_dyn_jac: Optional[Callable[[torch.Tensor, torch.Tensor, float],
            Tuple[torch.Tensor, torch.Tensor]]] = None,
            fd_eps: float = 1e-4,
            max_iter: int = 40,
            tol_x: float = 1e-6,
            tol_u: float = 1e-6,
            exit_unconverged: bool = False,
            detach_unconverged: bool = True,
            converge_tol: float = 1e-6,
            delta_u: Optional[float] = None,
            best_cost_eps: float = 1e-6,
            not_improved_lim: int = 10,
            verbose: int = 0
    ):
        super().__init__()
        # Dispositivo
        self.device = torch.device(device)

        # Dinamica e costi
        self.f_dyn = f_dyn
        self.total_time = total_time
        self.dt = step_size
        self.horizon = horizon
        self.cost_module = cost_module

        # Dimensioni di stato e controllo
        self.nx = cost_module.nx
        self.nu = cost_module.nu

        # Bound di controllo
        self.u_min = u_min.to(self.device) if u_min is not None else None
        self.u_max = u_max.to(self.device) if u_max is not None else None

        # Parametri solver
        self.reg_eps = reg_eps
        self.N_sim = N_sim if N_sim is not None else horizon

        # Metodo di derivazione
        if isinstance(grad_method, str):
            grad_method = GradMethod(grad_method.lower())
        self.grad_method = grad_method
        self.f_dyn_jac = f_dyn_jac
        self.fd_eps = fd_eps
        if self.grad_method is GradMethod.ANALYTIC and self.f_dyn_jac is None:
            raise ValueError("Per grad_method='analytic' serve f_dyn_jac(x,u,dt)->(A,B)")

        # Parametri linesearch, trust-region e convergenza
        self.delta_u = delta_u
        self.best_cost_eps = best_cost_eps
        self.not_improved_lim = not_improved_lim
        self.verbose = verbose

        # ---------- nuovi attributi ----------
        self.max_iter = int(max_iter)
        self.tol_u = float(tol_u)
        self.tol_x = float(tol_x)
        self.detach_unconverged = bool(detach_unconverged)
        self.converged: bool | None = None
        # Buffer per backward
        self.U_last = None
        self.X_last = None
        self.H_last = None
        self.F_last = None
        self.lmb_last = None
        self.tight_mask_last = None

        # Criteri di convergenza
        self.exit_unconverged = exit_unconverged
        self.detach_unconverged = detach_unconverged
        self.converge_tol = converge_tol
        self.converged = True

        # Warm-start
        self.U_prev = None

    #   Calcolo Jacobiane A, B
    def _jacobian_analytic(self, x: torch.Tensor, u: torch.Tensor):
        return self.f_dyn_jac(x, u, self.dt)

    def _jacobian_auto_diff(self, x: torch.Tensor, u: torch.Tensor):
        """Jacobian via autograd. Usa vectorize=True se disponibile."""
        A = torch.autograd.functional.jacobian(
            lambda xx: self.f_dyn(xx, u, self.dt), x,
            create_graph=True, vectorize=True
        )
        B = torch.autograd.functional.jacobian(
            lambda uu: self.f_dyn(x, uu, self.dt), u,
            create_graph=True, vectorize=True
        )
        return A, B

    ##########################DEBUG ONLY#############################################
    def _jacobian_finite_diff(self, x: torch.Tensor, u: torch.Tensor):
        """Central finite differences; non differenziabile ma più rapida di autograd.
        Restituisce A = d f / d x, B = d f / d u. DEBUG ONLY
        """
        fd_eps = self.fd_eps
        nx, nu = self.nx, self.nu
        f0 = self.f_dyn(x, u, self.dt)
        # ---- A --------------------------------------------------------
        eye_x = torch.eye(nx, device=x.device, dtype=x.dtype)
        A_cols = []
        for j in range(nx):
            dx = eye_x[j] * fd_eps
            f_plus = self.f_dyn(x + dx, u, self.dt)
            f_minus = self.f_dyn(x - dx, u, self.dt)
            A_cols.append(((f_plus - f_minus) / (2.0 * fd_eps)).unsqueeze(-1))
        A = torch.cat(A_cols, dim=-1)  # [nx, nx]
        # ---- B --------------------------------------------------------
        eye_u = torch.eye(nu, device=u.device, dtype=u.dtype)
        B_cols = []
        for j in range(nu):
            du = eye_u[j] * fd_eps
            f_plus = self.f_dyn(x, u + du, self.dt)
            f_minus = self.f_dyn(x, u - du, self.dt)
            B_cols.append(((f_plus - f_minus) / (2.0 * fd_eps)).unsqueeze(-1))
        B = torch.cat(B_cols, dim=-1)  # [nx, nu]
        return A, B

    def oldforward(
            self,
            x0: torch.Tensor,
            x_ref_full: torch.Tensor | None = None,
            u_ref_full: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Esegue il rollout completo del controllo MPC per N_sim passi, gestendo
        batch di traiettorie e riferimenti dinamici.
        """
        # 1) Normalizza la dimensione del batch per tutti gli input
        was_unbatched = x0.ndim == 1
        if was_unbatched:
            x0 = x0.unsqueeze(0)
        if x_ref_full is not None and x_ref_full.ndim == 2:
            x_ref_full = x_ref_full.unsqueeze(0)
        if u_ref_full is not None and u_ref_full.ndim == 2:
            u_ref_full = u_ref_full.unsqueeze(0)

        # Estrae le dimensioni
        B, device, dtype = x0.shape[0], x0.device, x0.dtype
        H, nu = self.horizon, self.nu

        # 2) Inizializza le variabili per il ciclo
        x_current = x0
        U_init = torch.zeros(B, H, nu, device=device, dtype=dtype)
        if getattr(self, 'U_prev', None) is not None and self.U_prev.shape[0] == B:
            U_init = torch.roll(self.U_prev, shifts=-1, dims=1)
            U_init[:, -1] = 0.0

        xs_list, us_list = [x0], []

        # 3) Ciclo di controllo MPC in anello chiuso
        for t in range(self.N_sim):
            x_ref_window = (
                x_ref_full[:, t: t + H + 1, :] if x_ref_full is not None else None
            )
            u_ref_window = (
                u_ref_full[:, t: t + H, :] if u_ref_full is not None else None
            )

            # Aggiorna i riferimenti nel modulo costi
            if x_ref_window is not None or u_ref_window is not None:
                self.cost_module.set_reference(
                    x_ref=x_ref_window,
                    u_ref=u_ref_window,
                )
            u_opt = ILQRSolve.apply(x_current, self, U_init)
            us_list.append(u_opt)
            x_current = self.f_dyn(x_current, u_opt, self.dt)
            xs_list.append(x_current)
            self.U_prev = self.U_last
            if self.U_last is not None:
                U_init = torch.roll(self.U_last, shifts=-1, dims=1)
                U_init[:, -1] = self.U_last[:, -1]

        # 4) Concatena i risultati in tensori
        Xs = torch.stack(xs_list, dim=1)
        Us = torch.stack(us_list, dim=1)

        if was_unbatched:
            Xs = Xs.squeeze(0)
            Us = Us.squeeze(0)

        return Xs, Us

    def forward(
            self,
            x0: Tensor,
            U_init: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Metodo forward principale. Prepara e chiama la funzione autograd ILQRSolve.
        """
        if U_init is None:
            U_init = torch.zeros(
                x0.shape[0], self.horizon, self.nu,
                device=x0.device, dtype=x0.dtype
            )

        # Prende i parametri di costo dal suo modulo interno per passarli a ILQRSolve.
        # Questo è il collegamento che permette ai gradienti di fluire indietro.
        C = self.cost_module.C
        c = self.cost_module.c
        C_final = self.cost_module.C_final
        c_final = self.cost_module.c_final

        # Chiama la nostra funzione autograd implementata manualmente
        return ILQRSolve.apply(x0, C, c, C_final, c_final, self, U_init)

    def solve_step(self, x0: Tensor, U_init: Tensor) -> Tuple[Tensor, Tensor]:
        """Risolve l'iLQR per l'intero batch in parallelo, senza cicli for."""
        B = x0.shape[0]
        U = U_init.clone()
        X = self.rollout_trajectory(x0, U)

        best_cost = self.cost_module.objective(X, U)

        for i in range(self.max_iter):
            # 1. Linearizza e quadratizza per l'intero batch in una sola volta
            A, Bm = self.linearize_dynamics(X, U)
            l_x, l_u, l_xx, l_xu, l_uu, l_xN, l_xxN = self.cost_module.quadraticize(X, U)

            # 2. Esegui il backward e forward pass in parallelo su tutto il batch
            bwd_pass_vmap = _vmap(self.backward_lqr)
            K, k = bwd_pass_vmap(A, Bm, l_x, l_u, l_xx, l_xu, l_uu, l_xN, l_xxN)

            fwd_pass_vmap = _vmap(self.forward_pass)
            X_new, U_new = fwd_pass_vmap(x0, X, U, K, k)

            # 3. Valuta e aggiorna le soluzioni migliori per ogni elemento del batch
            new_cost = self.cost_module.objective(X_new, U_new)

            improved_mask = new_cost < best_cost
            if not improved_mask.any():  # Se nessuno è migliorato, esci
                break

            best_cost = torch.where(improved_mask, new_cost, best_cost)

            update_mask_X = improved_mask.view(B, 1, 1)
            update_mask_U = improved_mask.view(B, 1, 1)

            X = torch.where(update_mask_X, X_new, X)
            U = torch.where(update_mask_U, U_new, U)

        self.converged = True  # Semplificato per ora

        # Salva i risultati finali, che verranno usati da ILQRSolve.backward
        self.H_last = (l_xx, l_uu, l_xu)
        self.F_last = (A, Bm)
        self.tight_mask_last = self._compute_tight_mask(U)  # Assicurati che questo metodo esista

        return X, U

    def forward_pass_batched(self, x0, X_ref, U_ref, K, k):
        """
        Esegue il forward pass per l'intero batch usando vmap.
        """

        def _forward_single(x0i, X_ref_i, U_ref_i, K_i, k_i):
            # Esegue il rollout per una singola traiettoria
            x_i = x0i
            xs = [x_i]
            us = []
            for t in range(self.horizon):
                dx_i = x_i - X_ref_i[t]
                du_i = k_i[t] + torch.einsum("ij,j->i", K_i[t], dx_i)  # K @ dx
                u_new = U_ref_i[t] + du_i

                # Applica i vincoli
                if self.u_min is not None:
                    u_new = torch.max(u_new, self.u_min)
                if self.u_max is not None:
                    u_new = torch.min(u_new, self.u_max)

                us.append(u_new)
                x_i = self.f_dyn(x_i, u_new, self.dt)
                xs.append(x_i)

            return torch.stack(xs, dim=0), torch.stack(us, dim=0)

        # Vettorizza la funzione di forward pass sul batch
        X_new, U_new = _vmap(_forward_single, in_dims=(0, 0, 0, 0, 0))(x0, X_ref, U_ref, K, k)
        return X_new, U_new


    def forward_pass(self, x0, X_ref, U_ref, K, k):
        X_new, U_new = [x0], []
        xt = x0
        for t in range(self.horizon):
            dx = xt - X_ref[t]
            du = K[t] @ dx + k[t]
            ut = U_ref[t] + du
            U_new.append(ut)
            xt = self.f_dyn(xt, ut, self.dt)
            X_new.append(xt)
        return torch.stack(X_new, dim=0), torch.stack(U_new, dim=0)

    def rollout_trajectory(
            self,
            x0: torch.Tensor,  # (B, nx)  **oppure** (nx,) per retro-compatibilità
            U: torch.Tensor  # (B, T, nu) **oppure** (T, nu)
    ) -> torch.Tensor:  # ritorna (B, T+1, nx)
        """
        Propaga la dinamica per tutti gli orizzonti in **parallelo sui batch**.

        Args
        ----
        x0 : (B, nx)      Stato iniziale batch (B può essere 1).
        U  : (B, T, nu)   Sequenza di comandi per batch.
                          Se passato shape (T, nu) verrà broadcastato su B=1.
        Returns
        -------
        X  : (B, T+1, nx) Traiettoria degli stati.
        """
        # -- normalizza le dimensioni ----------------------------------------
        if x0.ndim == 1:  # (nx,)  →  (1, nx)
            x0 = x0.unsqueeze(0)
        if U.ndim == 2:  # (T, nu) → (1, T, nu)
            U = U.unsqueeze(0)

        B, T, _ = U.shape
        nx = self.nx
        device, dtype = x0.device, x0.dtype

        # buffer trajectory
        X = torch.empty(B, T + 1, nx, device=device, dtype=dtype)
        X[:, 0] = x0
        xt = x0

        # loop temporale
        for t in range(T):
            ut = U[:, t]  # (B, nu)
            xt = self.f_dyn(xt, ut, self.dt)  # f_dyn deve supportare batch
            X[:, t + 1] = xt
        return X

    # -----------------------------------------------------------------

    def quadraticize_cost(self, X: Tensor, U: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Decompone i tensori di costo in blocchi x/u, preservando la dimensione del batch.
        Questo metodo è compatibile con input batched (B, T, ...) e unbatched (T, ...).
        """
        nx, nu = self.nx, self.nu
        is_batched = X.ndim == 3

        if is_batched:
            # --- MODIFICA: Passa i riferimenti a VMAP ---
            # Prende i riferimenti batchati dal modulo di costo
            x_ref_b = self.cost_module.x_ref
            u_ref_b = self.cost_module.u_ref

            # Assicura che i riferimenti abbiano lo stesso batch size di X
            B = X.shape[0]
            if B > 1 and x_ref_b.shape[0] == 1:
                x_ref_b = x_ref_b.expand(B, -1, -1)
            if B > 1 and u_ref_b.shape[0] == 1:
                u_ref_b = u_ref_b.expand(B, -1, -1)


            _quad_fn = lambda Xi, Ui, x_ref_i, u_ref_i: self.cost_module.quadraticize(
                Xi, Ui, x_ref_override=x_ref_i, u_ref_override=u_ref_i
            )

            # Passa i batch di traiettorie e riferimenti a vmap
            l_tau, H_tau, lN, HN = _vmap(_quad_fn)(X, U, x_ref_b, u_ref_b)

        else:
            # Il caso unbatched chiama direttamente la funzione (che userà self.x_ref)
            l_tau, H_tau, lN, HN = self.cost_module.quadraticize(X, U)


        l_x = l_tau[..., :, :nx]
        l_u = l_tau[..., :, nx:]
        l_xx = H_tau[..., :, :nx, :nx]
        l_xu = H_tau[..., :, :nx, nx:]
        l_uu = H_tau[..., :, nx:, nx:]
        l_xN = lN[..., :nx]
        l_xxN = HN[..., :nx, :nx]

        return l_x, l_u, l_xx, l_xu, l_uu, l_xN, l_xxN

    # -----------------------------------------------------------------
    """
    def linearize_dynamics_old(self, X: torch.Tensor, U: torch.Tensor):
        A_list, B_list = [], []
        for t in range(self.horizon):
            x = X[t].detach().clone().requires_grad_(self.grad_method is GradMethod.AUTO_DIFF)
            u = U[t].detach().clone().requires_grad_(self.grad_method is GradMethod.AUTO_DIFF)
            if self.grad_method is GradMethod.ANALYTIC:
                A_t, B_t = self._jacobian_analytic(x, u)
            elif self.grad_method is GradMethod.FINITE_DIFF:
                A_t, B_t = self._jacobian_finite_diff(x, u)
            else:
                A_t, B_t = self._jacobian_auto_diff(x, u)
            A_list.append(A_t)
            B_list.append(B_t)
        return torch.stack(A_list), torch.stack(B_list)
    """
    def linearize_dynamics(self, X: torch.Tensor, U: torch.Tensor):
        assert X.ndim == 3 and U.ndim == 3
        f = lambda x, u: self.f_dyn(x, u, self.dt)
        try:
            jac_x = jacrev(f, argnums=0)
            jac_u = jacrev(f, argnums=1)

            # _vmap(time) ∘ _vmap(batch)
            A = _vmap(_vmap(jac_x, in_dims=(0, 0)), in_dims=(0, 0))(X[:, :-1], U)
            B = _vmap(_vmap(jac_u, in_dims=(0, 0)), in_dims=(0, 0))(X[:, :-1], U)

            return A, B
        except RuntimeError as err:
            if self.debug:
                logging.warning(
                    "[linearize_dynamics] autograd-vmap failed – "
                    "switching to finite-diff.  Msg: %s", err)
            # X[:-1] perché l’ultima X è X_{T}
            A, B = jacobian_finite_diff_batched(
                self.f_dyn, X[:, :-1].reshape(-1, self.nx),  # (B·T, nx)
                U.reshape(-1, self.nu),  # (B·T, nu)
                dt=self.dt
            )
            # ri-shape per rimettere batch e tempo
            B_, T = X.shape[0], U.shape[1]
            A = A.reshape(B_, T, self.nx, self.nx)
            B = B.reshape(B_, T, self.nx, self.nu)
            return A, B
    # ------------------------------------------------------------------
    def backward_lqr(
            self,
            A: torch.Tensor,  # [T, nx, nx]
            B: torch.Tensor,  # [T, nx, nu]
            l_x: torch.Tensor,  # [T, nx]
            l_u: torch.Tensor,  # [T, nu]
            l_xx: torch.Tensor,  # [T, nx, nx]
            l_xu: torch.Tensor,  # [T, nx, nu]
            l_uu: torch.Tensor,  # [T, nu, nu]
            l_xN: torch.Tensor,  # [nx]
            l_xxN: torch.Tensor  # [nx, nx]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Riccati backward-pass con fallback robusto per Q_uu.
        Restituisce:
            K_seq : [T, nu, nx]
            k_seq : [T, nu]
        """
        T, nx, nu = A.shape[0], self.nx, self.nu
        dtype, device = A.dtype, A.device
        I_nu = torch.eye(nu, dtype=dtype, device=device) * self.reg_eps

        # FAST PATH con torch.func.scan
        if _HAS_SCAN:
            A_rev = torch.flip(A, dims=[0])
            B_rev = torch.flip(B, dims=[0])
            lx_rev = torch.flip(l_x, dims=[0])
            lu_rev = torch.flip(l_u, dims=[0])
            lxx_rev = torch.flip(l_xx, dims=[0])
            lxu_rev = torch.flip(l_xu, dims=[0])
            luu_rev = torch.flip(l_uu, dims=[0])

            def riccati_step(carry, inps):
                V, v = carry
                A_t, B_t, lx_t, lu_t, lxx_t, lxu_t, luu_t = inps

                # Q matrices
                Q_xx = lxx_t + A_t.T @ V @ A_t
                Q_xu = lxu_t + A_t.T @ V @ B_t
                Q_ux = Q_xu.mT
                Q_uu = luu_t + B_t.T @ V @ B_t + I_nu

                # rhs vectors
                q_x = lx_t + A_t.T @ v
                q_u = lu_t + B_t.T @ v

                # Inietta rumore per evitare le sing. se succede lo stesso usa la pseudo
                Q_uu_reg = Q_uu + self.reg_eps * torch.eye(nu, dtype=dtype, device=device)
                try:
                    L = torch.linalg.cholesky(Q_uu_reg)
                    K_t = -torch.cholesky_solve(Q_xu.mT, L)
                    k_t = -torch.cholesky_solve(q_u.unsqueeze(-1), L).squeeze(-1)
                except RuntimeError:
                    invQ = torch.linalg.pinv(Q_uu_reg)
                    K_t = -invQ @ Q_xu.mT
                    k_t = -invQ @ q_u

                #  cost-to-go
                V_new = Q_xx + K_t.T @ Q_uu @ K_t + K_t.T @ Q_ux + Q_xu @ K_t
                v_new = q_x + K_t.T @ Q_uu @ k_t + K_t.T @ q_u + Q_ux.mT @ k_t
                return (V_new, v_new), (K_t, k_t)

            _, (K_rev, k_rev) = scan(
                riccati_step,
                (l_xxN, l_xN),
                (A_rev, B_rev, lx_rev, lu_rev, lxx_rev, lxu_rev, luu_rev)
            )
            K_seq = torch.flip(K_rev, dims=[0])
            k_seq = torch.flip(k_rev, dims=[0])
            return K_seq, k_seq, None, None

        # FALLBACK: loop Python
        V = l_xxN
        v = l_xN
        K_list, k_list = [], []
        for t in reversed(range(T)):
            # Q matrices
            Q_xx = l_xx[t] + A[t].T @ V @ A[t]
            Q_xu = l_xu[t] + A[t].T @ V @ B[t]
            Q_ux = Q_xu.mT
            Q_uu = l_uu[t] + B[t].T @ V @ B[t] + I_nu

            # rhs
            q_x = l_x[t] + A[t].T @ v
            q_u = l_u[t] + B[t].T @ v
            Q_uu_reg = Q_uu + self.reg_eps * torch.eye(nu, dtype=dtype, device=device)
            try:
                L = torch.linalg.cholesky(Q_uu_reg)
                Kt = -torch.cholesky_solve(Q_xu.mT, L)
                kt = -torch.cholesky_solve(q_u.unsqueeze(-1), L).squeeze(-1)
            except RuntimeError:
                invQ = torch.linalg.pinv(Q_uu_reg)
                Kt = -invQ @ Q_xu.mT
                kt = -invQ @ q_u

            K_list.insert(0, Kt)
            k_list.insert(0, kt)

            # update cost-to-go
            V = Q_xx + Kt.T @ Q_uu @ Kt + Kt.T @ Q_ux + Q_xu @ Kt
            v = q_x + Kt.T @ Q_uu @ kt + Kt.T @ q_u + Q_ux.mT @ kt

        K_seq = torch.stack(K_list, dim=0)
        k_seq = torch.stack(k_list, dim=0)
        return K_seq, k_seq

    # -----------------------------------------------------------------
    def compute_cost(self, X: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        """
        Restituisce il costo batched (B,) SENZA cast a float.
        """
        return self.cost_module.objective(X, U)

    # -----------------------------------------------------------------
    def forward_with_linesearch(
            self,
            x0: torch.Tensor,  # [nx]
            X_ref: torch.Tensor,  # [T+1, nx]
            U_ref: torch.Tensor,  # [T,   nu]
            K: torch.Tensor,  # [T, nu, nx]
            k: torch.Tensor,  # [T, nu]
            alphas: Tuple[float, ...] = (1.0, 0.8, 0.6, 0.4, 0.2,
                                         0.1, 0.05, 0.01, 0.005, 0.0001)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Soluzione da cambiare in futuro con qualcosa di robusto
        Valuta in parallelo tutte le alpha restituisce la prima che riduce il costo,
        altrimenti la migliore. NB operazione in parallelo per migliorare l efficienza anche GPU
        """
        device, dtype = x0.device, x0.dtype
        nx, nu, T = self.nx, self.nu, self.horizon
        cost_ref = self.compute_cost(X_ref, U_ref)

        # prepara batch di alpha
        A = torch.tensor(alphas, dtype=dtype, device=device)[:, None]  # [A,1]
        n_alpha = A.size(0)

        # buffer per traiettorie parallele
        X_batch = torch.empty(n_alpha, T + 1, nx, dtype=dtype, device=device)
        U_batch = torch.empty(n_alpha, T, nu, dtype=dtype, device=device)
        X_batch[:, 0] = x0
        #  funzione dinamica in batch
        if _HAS_VMAP:
            f_dyn_batched = _vmap(lambda x, u: self.f_dyn(x, u, self.dt))
        else:  # fallback: con loop >_>
            def f_dyn_batched(x, u):
                return torch.stack([self.f_dyn(x[i], u[i], self.dt)
                                    for i in range(n_alpha)], dim=0)

        #  roll-out parallelo
        x = x0.expand(n_alpha, -1)  # [A, nx]
        for t in range(T):
            dx = x - X_ref[t]  # broadcast
            du = torch.matmul(dx, K[t].T) + A * k[t]  # [A, nu]
            u = U_ref[t] + du  # [A, nu]

            #  bounds
            if self.u_min is not None:
                u = torch.maximum(u, self.u_min)
            if self.u_max is not None:
                u = torch.minimum(u, self.u_max)

            U_batch[:, t, :] = u
            x = f_dyn_batched(x, u)  # [A, nx]
            X_batch[:, t + 1, :] = x

        costs = torch.tensor([self.compute_cost(X_batch[i], U_batch[i]) for i in range(n_alpha)], dtype=dtype,
                             device=device)
        improved = costs < cost_ref
        if improved.any():
            idx = int(torch.nonzero(improved, as_tuple=False)[0])
        else:
            idx = int(torch.argmin(costs))
        return X_batch[idx], U_batch[idx]
        # -----------------------------------------------------------------

    # -----------------------------------------------------------------
    def compute_costates(
            self,
            A: torch.Tensor,
            B: torch.Tensor,
            l_x: torch.Tensor,
            l_xN: torch.Tensor
    ) -> torch.Tensor:
        lambdas = [None] * (self.horizon + 1)
        lambdas[self.horizon] = l_xN
        for t in reversed(range(self.horizon)):
            lambdas[t] = A[t].T @ lambdas[t + 1] + l_x[t]
        return torch.stack(lambdas)

    # -----------------------------------------------------------------
    def _compute_tight_mask(self, U: torch.Tensor, atol: float = 1e-7) -> torch.Tensor:
        mask = torch.zeros_like(U, dtype=torch.bool)
        if self.u_min is not None:
            mask |= torch.isclose(U, self.u_min.expand_as(U), atol=atol)
        if self.u_max is not None:
            mask |= torch.isclose(U, self.u_max.expand_as(U), atol=atol)
        return mask

    # -----------------------------------------------------------------
    def eq8_gradients( #da eliminare
            self,
            grad_tau_x: torch.Tensor,
            grad_tau_u: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        dX, dU, _ = self.backward_pass_module2(grad_tau_x, grad_tau_u)
        Xs, Us = self.X_last, self.U_last
        Λs = self.lmb_last
        nx = self.nx
        gA = torch.zeros_like(self.F_last[0])
        gB = torch.zeros_like(self.F_last[1])
        gH_xx = torch.zeros_like(self.H_last[0])
        gH_uu = torch.zeros_like(self.H_last[1])
        gH_xu = torch.zeros_like(self.H_last[2])
        for t in range(self.horizon):
            tau_t = torch.cat([Xs[t], Us[t]])
            dtau_t = torch.cat([dX[t], dU[t]])
            dH_t = 0.5 * (self._outer(dtau_t, tau_t) + self._outer(tau_t, dtau_t))
            gH_xx[t] = dH_t[:nx, :nx]
            gH_uu[t] = dH_t[nx:, nx:]
            gH_xu[t] = dH_t[:nx, nx:]
            dF_t = self._outer(self.lmb_last[t + 1], tau_t) + self._outer(Λs[t + 1], dtau_t)
            gA[t] = dF_t[:, :nx]
            gB[t] = dF_t[:, nx:]
        return {"grad_x_init": dX[0], "grad_A": gA, "grad_B": gB,
                "grad_H_xx": gH_xx, "grad_H_uu": gH_uu, "grad_H_xu": gH_xu}

    # -----------------------------------------------------------------
    def _outer(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a.unsqueeze(-1) * b.unsqueeze(-2)

    # -----------------------------------------------------------------
    def backward_pass_module2(self, grad_tau_x: torch.Tensor, grad_tau_u: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calcola le variazioni dX, dU e dLambda per la sensibilità KKT
        usando il solver LQR con vincoli e trust-region su u_deltau.
        """
        # Hessiane e dinamiche dall’ultima risoluzione
        H_xx, H_uu, H_xu = self.H_last  # [T, nx, nx], [T, nu, nu], [T, nx, nu]
        A, B = self.F_last  # [T, nx, nx], [T, nx, nu]
        tight_mask = self.tight_mask_last  # [T, nu]

        #  LQR vincolata con PNQP
        dX, dU, _ = self._zero_constrained_lqr(
            A, B, H_xx, H_uu, H_xu,
            grad_tau_x[:-1],  # [T, nx]
            grad_tau_u,  # [T, nu]
            tight_mask,  # [T, nu]
            delta_u=self.delta_u  # trust-region
        )
        # dX: [T+1, nx], dU: [T, nu]

        # 2) calcolo dLambda (derivate dei cost-to-go) via backward recurrences
        dL = self._compute_dlambda(
            A, H_xx, H_xu,
            dX, dU,
            self.lmb_last[-1]  # lambda finale [nx]
        )
        # dL: [T+1, nx]
        return dX, dU, dL

    def _zero_constrained_lqr(
            self,
            A: torch.Tensor,  # [T, nx, nx]
            B: torch.Tensor,  # [T, nx, nu]
            H_xx: torch.Tensor,  # [T, nx, nx]
            H_uu: torch.Tensor,  # [T, nu, nu]
            H_xu: torch.Tensor,  # [T, nx, nu]
            grad_x: torch.Tensor,  # [T, nx]
            grad_u: torch.Tensor,  # [T, nu]
            tight_mask: torch.Tensor,  # [T, nu]
            delta_u: Optional[float] = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        T, nx, nu = A.shape[0], self.nx, self.nu
        dtype, device = A.dtype, A.device
        I_nu = torch.eye(nu, dtype=dtype, device=device) * self.reg_eps
        V = H_xx[-1]
        v = grad_x[-1]
        K_seq = [None] * T
        k_seq = [None] * T

        for t in reversed(range(T)):
            Q_xx = H_xx[t] + A[t].T @ V @ A[t]
            Q_xu = H_xu[t] + A[t].T @ V @ B[t]
            Q_ux = Q_xu.mT
            Q_uu = H_uu[t] + B[t].T @ V @ B[t] + I_nu

            q_x = grad_x[t] + A[t].T @ v
            q_u = grad_u[t] + B[t].T @ v

            # Solve constrained QP on du
            free = ~tight_mask[t]
            if free.any():
                lb = torch.full((nu,), -float('inf'), device=device, dtype=dtype)
                ub = torch.full((nu,), float('inf'), device=device, dtype=dtype)
                if self.u_min is not None:
                    lb = self.u_min - self.U_last[t]
                    ub = self.u_max - self.U_last[t]
                if delta_u is not None:
                    lb = torch.maximum(lb, torch.full_like(lb, -delta_u))
                    ub = torch.minimum(ub, torch.full_like(ub, delta_u))

                H_f = Q_uu[free][:, free]
                q_f = q_u[free]
                lb_f = lb[free]
                ub_f = ub[free]

                du_batch, diagnostics = pnqp(
                    H_f.unsqueeze(0),
                    q_f.unsqueeze(0),
                    lb_f.unsqueeze(0),
                    ub_f.unsqueeze(0)
                )
                du_f = du_batch.squeeze(0)
                du = torch.zeros(nu, dtype=dtype, device=device)
                du[free] = du_f
            else:
                du = torch.zeros(nu, dtype=dtype, device=device)

            # Feedback gains (unconstrained part)
            Kt = -torch.linalg.solve(Q_uu + 1e-8 * I_nu, Q_ux)
            kt = du
            K_seq[t] = Kt
            k_seq[t] = kt

            V = Q_xx + Kt.T @ Q_uu @ Kt + Kt.T @ Q_ux + Q_xu @ Kt
            v = q_x + Kt.T @ Q_uu @ kt + Kt.T @ q_u + Q_ux.mT @ kt

        K_seq = torch.stack(K_seq, dim=0)
        k_seq = torch.stack(k_seq, dim=0)

        # Forward pass: compute dX, dU
        dX = [torch.zeros(self.nx, dtype=dtype, device=device)]
        dU = []
        for t in range(T):
            dx = dX[-1]
            du = K_seq[t] @ dx + k_seq[t]
            dU.append(du)
            dX.append(A[t] @ dx + B[t] @ du)

        dX = torch.stack(dX, dim=0)
        dU = torch.stack(dU, dim=0)
        return dX, dU, K_seq

    # -----------------------------------------------------------------

    def reset(self) -> None:
        """
        Resetta lo stato interno del controller memorizzato dall'ultima esecuzione workaround per l'uso in cicli di training dove .backward()
        viene chiamato ripetutamente.
        """
        if self.verbose > 0:
            print("Resetting MPC controller internal state.")

        self.U_last = None
        self.X_last = None
        self.H_last = None
        self.F_last = None
        self.lmb_last = None
        self.tight_mask_last = None
        self.converged = None

    def _compute_dlambda(
            self,
            A: torch.Tensor,
            H_xx: torch.Tensor,
            H_xu: torch.Tensor,
            dX: torch.Tensor,
            dU: torch.Tensor,
            lmb_T: torch.Tensor
    ) -> torch.Tensor:
        T, nx = self.horizon, self.nx
        dL = [None] * (T + 1)
        dL[T] = H_xx[-1] @ dX[-1]
        for t in reversed(range(T)):
            term = H_xx[t] @ dX[t] + H_xu[t] @ dU[t]
            dL[t] = A[t].T @ dL[t + 1] + term
        return torch.stack(dL)

    def tracking_mpc(  # not a clever implementation
            self,
            x0: torch.Tensor,
            x_ref_full: torch.Tensor,
            u_ref_full: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nx, nu, H = self.nx, self.nu, self.horizon
        N_sim = x_ref_full.size(0) - 1
        assert x_ref_full.shape == (N_sim + 1, nx)

        if u_ref_full is None:
            u_ref_full = torch.zeros(N_sim, nu, dtype=x_ref_full.dtype,
                                     device=x_ref_full.device)
        else:
            assert u_ref_full.shape == (N_sim, nu)

        Xs = torch.empty(N_sim + 1, nx, dtype=x_ref_full.dtype,
                         device=x_ref_full.device)
        Us = torch.empty(N_sim, nu, dtype=x_ref_full.dtype,
                         device=x_ref_full.device)

        Xs[0] = x = x0.to(x_ref_full)
        U_prev = torch.zeros(H, nu, dtype=x_ref_full.dtype,
                             device=x_ref_full.device)

        for k in range(N_sim):
            xr_win = x_ref_full[k: k + H + 1]
            ur_win = u_ref_full[k: k + H]
            self.cost_module.set_reference(xr_win, ur_win)

            u = self.solve_step(x, U_prev)
            Us[k] = u
            x = self.f_dyn(x, u, self.dt)
            Xs[k + 1] = x
            U_prev = self.U_last

        return Xs, Us