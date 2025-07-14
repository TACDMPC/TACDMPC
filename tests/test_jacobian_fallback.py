import torch
import pytest
from DifferentialMPC.controller import DifferentiableMPCController
from DifferentialMPC.cost import GeneralQuadCost
from DifferentialMPC import GradMethod

# ------------------- piccola dinamica helper ---------------------------------
def f_simple(x, u, dt):
    """Dinamica lineare x' = x + dt·u (nx=1, nu=1)"""
    return x + dt * u

def f_simple_nondiff(x, u, dt):
    """Stessa dinamica MA con round() non-differenziabile → forza il fallback"""
    return torch.round(x + dt * u)     # torch.round ha grad non definito


# ------------------- test parametrico ----------------------------------------
@pytest.mark.parametrize("nondiff", [False, True])
def test_linearize_dynamics_fallback(nondiff):
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- MPC minimale ------------------------------------------------------
    nx = nu = 1
    H  = 5
    dt = 0.1

    dyn = f_simple_nondiff if nondiff else f_simple

    cost = GeneralQuadCost(
        nx=nx, nu=nu,
        C=torch.zeros(H, nx + nu, nx + nu, device=device),
        c=torch.zeros(H, nx + nu,              device=device),
        C_final=torch.zeros(nx + nu, nx + nu,  device=device),
        c_final=torch.zeros(nx + nu,           device=device),
        device=str(device),
    )

    mpc = DifferentiableMPCController(
        f_dyn=dyn,
        total_time=H * dt,
        step_size=dt,
        horizon=H,
        cost_module=cost,
        grad_method=GradMethod.AUTO_DIFF,   # proverà jacrev, poi fallback se serve
        device=str(device)                       # per vedere il warning, opzionale
    )

    # ------- batch input -----------------------------------------------------
    BATCH = 4
    x0 = torch.zeros(BATCH, nx, device=device, requires_grad=True)

    # ------- forward & backward ---------------------------------------------
    X_opt, U_opt = mpc.forward(x0)
    loss = X_opt[:, -1].pow(2).sum()
    loss.backward()                      # deve passare in entrambi i casi

    # grad su x0 deve esistere
    assert x0.grad is not None
    assert torch.isfinite(x0.grad).all()
