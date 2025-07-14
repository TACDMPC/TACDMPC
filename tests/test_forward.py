# tests/test_cartpole_mpc.py
import torch
from dataclasses import dataclass
import gym, os
from contextlib import redirect_stdout

from DifferentialMPC import DifferentiableMPCController, GeneralQuadCost, GradMethod


# ----------------------------------------------------------------------
# 1) Dinamica Cart-Pole (batch-friendly)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class CartPoleParams:
    m_c: float; m_p: float; l: float; g: float

    @classmethod
    def from_gym(cls):
        # sopprime la stampa d’inizializzazione Gym
        with open(os.devnull, "w") as f, redirect_stdout(f):
            env = gym.make("CartPole-v1")
        return cls(float(env.unwrapped.masscart),
                   float(env.unwrapped.masspole),
                   float(env.unwrapped.length),
                   float(env.unwrapped.gravity))


def f_cartpole(x: torch.Tensor, u: torch.Tensor, dt: float,
               p: CartPoleParams) -> torch.Tensor:
    """dynamics:  (B,4), (B,1)  ->  (B,4)"""
    pos, vel, theta, omega = torch.unbind(x, dim=-1)
    force = u.squeeze(-1)

    sin_t, cos_t = torch.sin(theta), torch.cos(theta)
    tot_mass, m_p_l = p.m_c + p.m_p, p.m_p * p.l

    temp     = (force + m_p_l * omega ** 2 * sin_t) / tot_mass
    theta_dd = (p.g * sin_t - cos_t * temp) / (p.l * (4/3 - p.m_p * cos_t ** 2 / tot_mass))
    vel_dd   = temp - m_p_l * theta_dd * cos_t / tot_mass

    return torch.stack((pos + vel * dt,
                        vel + vel_dd * dt,
                        theta + omega * dt,
                        omega + theta_dd * dt), dim=-1)


# Jacobiana opzionale (serve se grad_method="analytic")
def f_cartpole_jac(x: torch.Tensor, u: torch.Tensor, dt: float):
    return torch.autograd.functional.jacobian(
        lambda xx: f_cartpole(xx, u, dt, params), x, create_graph=True, vectorize=True
    ), torch.autograd.functional.jacobian(
        lambda uu: f_cartpole(x, uu, dt, params), u, create_graph=True, vectorize=True
    )


# ----------------------------------------------------------------------
# 2) Test veri e propri
# ----------------------------------------------------------------------
def test_batch_cartpole_mpc_forward_backward():
    torch.set_default_dtype(torch.float64)
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH    = 8
    DT, H, N = 0.05, 20, 150
    nx, nu   = 4, 1

    # --- dinamica & costi ------------------------------------------------
    global params           # così resta visibile a f_cartpole_jac
    params = CartPoleParams.from_gym()
    dyn = lambda x, u, dt: f_cartpole(x, u, dt, params)

    Q = torch.diag(torch.tensor([10., 1., 100., 1.], device=device))
    R = torch.diag(torch.tensor([0.1], device=device))
    C = torch.zeros(H, nx+nu, nx+nu, device=device); C[:, :nx, :nx] = Q; C[:, nx:, nx:] = R
    c = torch.zeros_like(C[..., 0])
    C_final, c_final = C[0]*10, torch.zeros(nx+nu, device=device)
    cost = GeneralQuadCost(nx, nu, C, c, C_final, c_final, device=str(device))

    mpc = DifferentiableMPCController(
        f_dyn=dyn, total_time=H*DT, step_size=DT, horizon=H, cost_module=cost,
        u_min=torch.tensor([-20.], device=device),
        u_max=torch.tensor([ 20.], device=device),
        grad_method=GradMethod.AUTO_DIFF,   # se preferisci analytic -> passa jacobiana
        N_sim=N, device=str(device)
    )

    # --- batch di input ---------------------------------------------------
    x0 = (torch.tensor([0., 0., 0.2, 0.], device=device).repeat(BATCH, 1).requires_grad_())
    x_ref = torch.zeros(BATCH, N+H+1, nx, device=device)   # inseguimento a 0
    u_ref = torch.zeros(BATCH, N+H  , nu, device=device)

    # --- forward ----------------------------------------------------------
    X_star, U_star = mpc.forward(x0, x_ref_full=x_ref, u_ref_full=u_ref)

    assert X_star.shape == (BATCH, N + 1, nx)
    assert U_star.shape == (BATCH, N, nu)

    # --- verifica backward -----------------------------------------------
    # perdiamo semplice: sommiamo tutti gli stati finali e facciamo backward
    loss = X_star[:, -1].pow(2).sum()
    loss.backward()            # deve passare senza errori/graffiti
