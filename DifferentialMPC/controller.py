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

class ILQRSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                x0: Tensor,
                # Parametri e riferimenti sono input espliciti per la differenziabilità
                C: Tensor, c: Tensor, C_final: Tensor, c_final: Tensor,
                x_ref: Tensor, u_ref: Tensor,
                controller: 'DifferentiableMPCController',
                U_init: Tensor
                ) -> Tuple[Tensor, Tensor]:

        # 1. Assegna parametri e riferimenti al modulo del controller
        controller.cost_module.C = C
        controller.cost_module.c = c
        controller.cost_module.C_final = C_final
        controller.cost_module.c_final = c_final
        controller.cost_module.set_reference(x_ref, u_ref) # Ora fa parte del grafo

        # 2. Risolvi l'MPC
        X_opt, U_opt = controller.solve_step(x0, U_init)

        # 3. Stacca se non convergente
        if (not controller.converged) and controller.detach_unconverged:
            X_opt, U_opt = X_opt.detach(), U_opt.detach()

        # 4. Salva i tensori per il backward pass
        ctx.controller = controller
        ctx.save_for_backward(
            X_opt, U_opt,
            controller.H_last[0], controller.H_last[1], controller.H_last[2],
            controller.F_last[0], controller.F_last[1],
            controller.tight_mask_last,
            x_ref, u_ref  # Salva anche i riferimenti
        )

        return X_opt, U_opt


    @staticmethod
    def backward(ctx, grad_X_out: Tensor, grad_U_out: Tensor):
        # 1. Recupera i dati salvati
        X, U, H_xx, H_uu, H_xu, A, Bm, tight_mask, x_ref, u_ref = ctx.saved_tensors
        ctrl = ctx.controller
        B, T, nx, nu = X.shape[0], U.shape[1], ctrl.nx, ctrl.nu

        # 2. Risolvi l'LQR all'indietro per ottenere le sensitività (dX/dθ, dU/dθ)
        dX_list, dU_list = [], []
        for i in range(B):
            dX_i, dU_i, _ = ctrl._zero_constrained_lqr(
                A[i], Bm[i], H_xx[i], H_uu[i], H_xu[i],
                -grad_X_out[i, :-1], -grad_U_out[i], tight_mask[i],U[i]
            )
            dX_list.append(dX_i)
            dU_list.append(dU_i)
        dX = torch.stack(dX_list)
        dU = torch.stack(dU_list)

        # 3. Calcola i gradienti dei parametri di costo
        # Errore e derivata dell'errore
        err_x, err_u = X[:, :-1] - x_ref[:, :T], U - u_ref
        derr_x, derr_u = dX[:, :-1], dU

        # Gradiente per costi di tappa (running costs)
        tau = torch.cat([err_x, err_u], dim=-1)
        dtau = torch.cat([derr_x, derr_u], dim=-1)
        grad_C = -0.5 * (_outer(dtau, tau) + _outer(tau, dtau))
        grad_c = -dtau

        # Gradiente per costo finale (terminal cost)
        err_xN = X[:, -1] - x_ref[:, -1]
        derr_xN = dX[:, -1]
        grad_C_final_xx = -0.5 * (_outer(derr_xN, err_xN) + _outer(err_xN, derr_xN))
        grad_C_final = torch.nn.functional.pad(grad_C_final_xx, (0, nu, 0, nu))
        grad_c_final = torch.nn.functional.pad(-derr_xN, (0, nu))

        # 4. Calcola i gradienti degli input originali
        grad_x0 = dX[:, 0]
        # dL/dx_ref = (dL/d_err_x) * (d_err_x / dx_ref) = (-derr_x) * (-1) = derr_x
        grad_xref = torch.zeros_like(x_ref)
        grad_xref[:, :T, :] = derr_x
        grad_xref[:, -1, :] = derr_xN
        grad_uref = derr_u

        # 5. Restituisci i gradienti nell'ordine corretto degli input di forward()
        return (
            grad_x0,
            grad_C, grad_c, grad_C_final, grad_c_final,
            grad_xref, grad_uref,
            None,  # grad per controller
            None  # grad per U_init
        )


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

    def forward(self, x0: Tensor, U_init: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        B = x0.shape[0] if x0.ndim > 1 else 1
        if U_init is None:
            U_init = torch.zeros(B, self.horizon, self.nu, device=x0.device, dtype=x0.dtype)
        C, c, C_final, c_final = self.cost_module.C, self.cost_module.c, self.cost_module.C_final, self.cost_module.c_final
        x_ref, u_ref = self.cost_module.x_ref, self.cost_module.u_ref

        return ILQRSolve.apply(x0, C, c, C_final, c_final, x_ref, u_ref, self, U_init)

    def solve_step(self, x0: Tensor, U_init: Tensor) -> Tuple[Tensor, Tensor]:
        B = x0.shape[0]
        U, X = U_init.clone(), self.rollout_trajectory(x0, U_init)

        x_ref_batch, u_ref_batch = self.cost_module.x_ref, self.cost_module.u_ref
        best_cost = self.cost_module.objective(X, U, x_ref_override=x_ref_batch, u_ref_override=u_ref_batch)

        for i in range(self.max_iter):
            l_x, l_u, l_xx, l_xu, l_uu, l_xN, l_xxN = self.cost_module.quadraticize(X, U)
            A, Bm = self.linearize_dynamics(X, U)

            K, k = _vmap(self.backward_lqr)(A, Bm, l_x, l_u, l_xx, l_xu, l_uu, l_xN, l_xxN)

            # restituisce  i candidati e i loro costi
            X_candidates, U_candidates, candidate_costs = _vmap(self.evaluate_alphas, in_dims=(0, 0, 0, 0, 0, 0, 0))(
                x0, X, U, K, k, x_ref_batch, u_ref_batch
            )
            best_alpha_indices = torch.argmin(candidate_costs, dim=1)
            idx_x = best_alpha_indices.view(B, 1, 1, 1).expand(-1, 1, X.shape[1], X.shape[2])
            X_new = torch.gather(X_candidates, 1, idx_x).squeeze(1)

            idx_u = best_alpha_indices.view(B, 1, 1, 1).expand(-1, 1, U.shape[1], U.shape[2])
            U_new = torch.gather(U_candidates, 1, idx_u).squeeze(1)
            new_cost = self.cost_module.objective(X_new, U_new, x_ref_override=x_ref_batch, u_ref_override=u_ref_batch)

            improved_mask = new_cost < best_cost
            if not improved_mask.any(): break

            best_cost = torch.where(improved_mask, new_cost, best_cost)
            X = torch.where(improved_mask.view(B, 1, 1), X_new, X)
            U = torch.where(improved_mask.view(B, 1, 1), U_new, U)

            self.cost_module.set_reference(x_ref_batch, u_ref_batch)

        self.converged = True
        self.cost_module.set_reference(x_ref_batch, u_ref_batch)
        l_x, l_u, l_xx, l_xu, l_uu, l_xN, l_xxN = self.cost_module.quadraticize(X, U)
        A, Bm = self.linearize_dynamics(X, U)
        self.H_last, self.F_last = (l_xx, l_uu, l_xu), (A, Bm)
        self.tight_mask_last, self.X_last, self.U_last = self._compute_tight_mask(U), X, U
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

    def linearize_dynamics(self, X: Tensor, U: Tensor):
        B, T, nx, nu = X.shape[0], U.shape[1], self.nx, self.nu

        if self.grad_method is GradMethod.AUTO_DIFF and _HAS_VMAP:
            f = lambda x, u: self.f_dyn(x, u, self.dt)
            jac_x, jac_u = jacrev(f, argnums=0), jacrev(f, argnums=1)
            A = _vmap(_vmap(jac_x, in_dims=(0, 0)), in_dims=(0, 0))(X[:, :-1], U)
            B = _vmap(_vmap(jac_u, in_dims=(0, 0)), in_dims=(0, 0))(X[:, :-1], U)
            return A, B
        elif self.grad_method is GradMethod.ANALYTIC:
            x_flat = X[:, :-1].reshape(-1, nx)  # Shape: [B*T, nx]
            u_flat = U.reshape(-1, nu)  # Shape: [B*T, nu]
            A_flat, B_flat = self.f_dyn_jac(x_flat, u_flat, self.dt)
            A = A_flat.reshape(B, T, nx, nx)
            B = B_flat.reshape(B, T, nx, nu)
            return A, B

        # Fallback per differenze finite
        else:
            print("attenzione il metodo non e differenziabile e non e adatto per RL")
            A, B = jacobian_finite_diff_batched(
                self.f_dyn, X[:, :-1].reshape(-1, self.nx), U.reshape(-1, self.nu), dt=self.dt
            )
            return A.reshape(B, T, nx, nx), B.reshape(B, T, nx, nu)

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

    def evaluate_alphas(
            self, x0: Tensor, X_ref: Tensor, U_ref: Tensor, K: Tensor, k: Tensor,
            x_ref_traj: Tensor, u_ref_traj: Tensor,
            alphas: Tuple[float, ...] = (1.0, 0.8, 0.5, 0.2, 0.1)
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Calcola le traiettorie e i costi per tutti gli alpha in parallelo.
        Restituisce tutti i candidati, non solo il migliore.
        """
        A = torch.tensor(alphas, dtype=x0.dtype, device=x0.device)
        n_alpha = A.shape[0]
        x_current_batch = x0.expand(n_alpha, -1)

        xs_list, us_list = [x_current_batch], []
        for t in range(self.horizon):
            dx_batch = x_current_batch - X_ref[t]
            du_batch = A.view(-1, 1) * k[t] + torch.einsum('ij,aj->ai', K[t], dx_batch)
            u_batch = U_ref[t] + du_batch
            if self.u_min is not None: u_batch = torch.max(u_batch, self.u_min)
            if self.u_max is not None: u_batch = torch.min(u_batch, self.u_max)
            us_list.append(u_batch)
            x_current_batch = self.f_dyn(x_current_batch, u_batch, self.dt)
            xs_list.append(x_current_batch)

        X_candidates = torch.stack(xs_list, dim=1)  # Shape [n_alpha, H+1, nx]
        U_candidates = torch.stack(us_list, dim=1)  # Shape [n_alpha, H, nu]

        objective_fn = lambda x, u: self.cost_module.objective(x, u, x_ref_override=x_ref_traj,
                                                               u_ref_override=u_ref_traj)
        candidate_costs = _vmap(objective_fn)(X_candidates, U_candidates)  # Shape [n_alpha]

        return X_candidates, U_candidates, candidate_costs

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
            U_last_i: torch.Tensor,
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
                    lb = self.u_min - U_last_i[t]
                    ub = self.u_max - U_last_i[t]
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