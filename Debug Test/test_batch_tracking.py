import sys
import os
import torch
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from DifferentialMPC import DifferentiableMPCController, GeneralQuadCost


# Definiamo la dinamica lineare direttamente qui, perché serve solo per questo test
def f_dyn_linear(x, u, dt):
    A = torch.tensor([[1.0, dt], [0.0, 1.0]], dtype=x.dtype, device=x.device)
    B = torch.tensor([[0.0], [dt]], dtype=x.dtype, device=x.device)
    return torch.einsum("...ij,...j->...i", A, x) + torch.einsum("...ij,...j->...i", B, u)


def f_dyn_jac_linear(x, u, dt):
    A = torch.tensor([[1.0, dt], [0.0, 1.0]], dtype=x.dtype, device=x.device)
    B = torch.tensor([[0.0], [dt]], dtype=x.dtype, device=x.device)
    return A, B


def test_end_to_end_batch_tracking():
    """
    Verifica end-to-end che il controller MPC gestisca correttamente
    riferimenti distinti per ogni elemento di un batch.
    """
    # --- SETUP ---
    DEVICE = "cpu"
    DTYPE = torch.float64
    torch.set_default_dtype(DTYPE)

    nx, nu = 2, 1
    T = 5  # Orizzonte MPC
    N_sim = 20  # Passi di simulazione
    dt = 0.1
    BATCH_SIZE = 2

    # --- Costo ---
    Q = torch.diag(torch.tensor([1.0, 0.1], device=DEVICE))
    R = torch.diag(torch.tensor([0.01], device=DEVICE))
    C = torch.zeros(T, nx + nu, nx + nu, device=DEVICE);
    C[:, :nx, :nx] = Q;
    C[:, nx:, nx:] = R
    c = torch.zeros(T, nx + nu, device=DEVICE)
    C_final = C[0].clone() * 10
    c_final = torch.zeros(nx + nu, device=DEVICE)
    cost_module = GeneralQuadCost(nx, nu, C, c, C_final, c_final, device=DEVICE)

    # --- MPC Controller ---
    mpc = DifferentiableMPCController(
        f_dyn=f_dyn_linear, total_time=T * dt, step_size=dt, horizon=T,
        cost_module=cost_module, grad_method="analytic", f_dyn_jac=f_dyn_jac_linear,
        device=DEVICE, N_sim=N_sim
    )

    # --- LOGICA DEL TEST ---
    print("Esecuzione test di batch tracking...")

    # Stati iniziali distinti per i due agenti del batch
    x0_A = torch.tensor([4.0, 0.0], device=DEVICE, dtype=DTYPE)
    x0_B = torch.tensor([-4.0, 0.0], device=DEVICE, dtype=DTYPE)
    x0_batch = torch.stack([x0_A, x0_B], dim=0)

    # Traiettorie di riferimento distinte
    ref_len = N_sim + T + 1
    time_steps = torch.linspace(0, 10, ref_len, device=DEVICE, dtype=DTYPE)
    x_ref_A = torch.stack([torch.sin(time_steps), torch.cos(time_steps)], dim=1)
    x_ref_B = torch.stack([0.5 * torch.cos(time_steps), -0.5 * torch.sin(time_steps)], dim=1)
    x_ref_full_batch = torch.stack([x_ref_A, x_ref_B], dim=0)
    u_ref_full_batch = torch.zeros(BATCH_SIZE, ref_len - 1, nu, device=DEVICE, dtype=DTYPE)

    # Esecuzione del forward pass del controller
    Xs, Us = mpc.forward(
        x0_batch,
        x_ref_full=x_ref_full_batch,
        u_ref_full=u_ref_full_batch
    )

    # --- VERIFICHE ---
    # 1. L'output deve avere la forma corretta
    assert Xs.shape == (BATCH_SIZE, N_sim + 1, nx)

    # 2. Le traiettorie risultanti devono essere diverse, altrimenti il
    #    controller ha ignorato i riferimenti distinti.
    assert not torch.allclose(Xs[0], Xs[1]), "Le traiettorie sono identiche! Il test è fallito."

    print("✅ Test di batch tracking superato con successo!")


if __name__ == "__main__":
    test_end_to_end_batch_tracking()