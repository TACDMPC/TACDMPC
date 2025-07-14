import torch
import pytest
from DifferentialMPC.controller import DifferentiableMPCController
from DifferentialMPC.cost import GeneralQuadCost
from DifferentialMPC import GradMethod

@pytest.mark.parametrize("fallback", [False, True])
def test_gradcheck_mpc(fallback):
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.double)          # gradcheck richiede double

    # --------- dinamica 1D minimal -----------------------------------------
    def f_lin(x, u, dt):         # grad-safe
        return x + dt * u
    def f_round(x, u, dt):       # non grad-safe → fallback
        return torch.round(x + dt * u)

    f_dyn = f_round if fallback else f_lin

    nx = nu = 1
    H  = 4
    dt = 0.1

    cost = GeneralQuadCost(
        nx, nu,
        C=torch.zeros(H, nx+nu, nx+nu, device=device, dtype=torch.double),
        c=torch.zeros(H, nx+nu,              device=device, dtype=torch.double),
        C_final=torch.zeros(nx+nu, nx+nu,    device=device, dtype=torch.double),
        c_final=torch.zeros(nx+nu,           device=device, dtype=torch.double),
        device=str(device),
    )

    mpc = DifferentiableMPCController(
        f_dyn=f_dyn,
        total_time=H*dt,
        step_size=dt,
        horizon=H,
        cost_module=cost,
        grad_method=GradMethod.AUTO_DIFF,    # proverà autograd → fallback se serve
        device=str(device),
    )

    # gradcheck vuole input double con requires_grad=True
    x0 = torch.randn(1, nx, device=device, dtype=torch.double, requires_grad=True)

    def func(x):
        X_opt, _ = mpc.forward(x)    # prendiamo solo gli stati
        return X_opt.sum()           # scalar output

    assert torch.autograd.gradcheck(func, (x0,), eps=1e-6, atol=1e-4, rtol=1e-3)
