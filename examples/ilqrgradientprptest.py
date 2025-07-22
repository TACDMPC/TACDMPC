# ilqrgradientprptest.py (Corretto)

import torch
import numpy as np

# Assicurati che i file siano nella stessa directory o nel python path
from DifferentialMPC import DifferentiableMPCController, GradMethod
from DifferentialMPC import GeneralQuadCost

# --- Definizione di un ActorMPC fittizio per il test ---
class MockActorMPC(torch.nn.Module):
    def __init__(self, nx, nu, horizon, dt, f_dyn, f_dyn_jac, device):
        super().__init__()
        self.device = device
        self.n_tau = nx + nu
        T = horizon

        cost_out_dim = (T * self.n_tau**2) + (T * self.n_tau) + \
                       (self.n_tau**2) + self.n_tau
        self.cost_map_net = torch.nn.Sequential(
            torch.nn.Linear(nx, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, cost_out_dim)
        ).to(device)

        cost_module = GeneralQuadCost(
            nx, nu, C=torch.zeros(T, self.n_tau, self.n_tau, device=device),
            c=torch.zeros(T, self.n_tau, device=device),
            C_final=torch.zeros(self.n_tau, self.n_tau, device=device),
            c_final=torch.zeros(self.n_tau, device=device),
            device=device
        )

        # Controller MPC
        self.mpc_controller = DifferentiableMPCController(
            f_dyn=f_dyn,
            # <<< MODIFICA: Reintrodotto il parametro 'total_time' richiesto >>>
            total_time=horizon * dt,
            step_size=dt,
            horizon=horizon,
            cost_module=cost_module,
            device=device,
            grad_method=GradMethod.ANALYTIC,
            f_dyn_jac=f_dyn_jac
        )

    def forward(self, x0_batch):
        B = x0_batch.shape[0]
        T, nx, nu, n_tau = self.mpc_controller.horizon, self.mpc_controller.nx, self.mpc_controller.nu, self.n_tau

        # 1. La rete produce un vettore di parametri di costo
        cost_params = self.cost_map_net(x0_batch)

        # 2. Riorganizza il vettore nei tensori C, c, C_final, c_final
        C_flat, c_flat, C_final_flat, c_final_flat = torch.split(cost_params, [
            T * n_tau ** 2, T * n_tau, n_tau ** 2, n_tau
        ], dim=-1)

        C = C_flat.reshape(B, T, n_tau, n_tau)
        c = c_flat.reshape(B, T, n_tau)
        C_final = C_final_flat.reshape(B, n_tau, n_tau)
        c_final = c_final_flat.reshape(B, n_tau)

        # 3. Aggiorna il modulo di costo del controller con i parametri appresi
        self.mpc_controller.cost_module.C = C
        self.mpc_controller.cost_module.c = c
        self.mpc_controller.cost_module.C_final = C_final
        self.mpc_controller.cost_module.c_final = c_final

        # <<< MODIFICA CHIAVE E DEFINITIVA >>>
        # Creiamo e impostiamo dei riferimenti fittizi con la dimensione di batch corretta.
        # Questo allinea le dimensioni di tutti gli input per vmap.
        x_ref_dummy = torch.zeros(B, T + 1, nx, device=self.device, dtype=x0_batch.dtype)
        u_ref_dummy = torch.zeros(B, T, nu, device=self.device, dtype=x0_batch.dtype)
        self.mpc_controller.cost_module.set_reference(x_ref_dummy, u_ref_dummy)
        # <<< FINE MODIFICA >>>

        # 4. Chiama il forward del controller
        X_opt, U_opt = self.mpc_controller.forward(x0_batch)

        return X_opt, U_opt


print("=" * 60)
print("🧪 ESECUZIONE TEST DI PROPAGAZIONE DEL GRADIENTE 🧪")
print("=" * 60)

# --- Parametri di base per il test ---
NX, NU, HORIZON, DT = 2, 1, 5, 0.1
DEVICE = "cpu"
BATCH_SIZE = 4

# --- Dinamica e Jacobiana ---
def f_dyn_batched(x, u, dt=DT):
    p, v = x.split(1, dim=-1)
    return torch.cat([p + v * dt, v + u * dt], dim=-1)

def f_dyn_jac_unbatched(x, u, dt=DT):
    A = torch.tensor([[1, dt], [0, 1]], device=x.device, dtype=x.dtype)
    B_mat = torch.tensor([[0], [dt]], device=x.device, dtype=x.dtype)
    return A, B_mat

# --- Esecuzione del Test ---
try:
    actor = MockActorMPC(
        nx=NX, nu=NU, horizon=HORIZON, dt=DT,
        f_dyn=f_dyn_batched,
        f_dyn_jac=f_dyn_jac_unbatched,
        device=DEVICE
    )
    test_input = torch.randn(BATCH_SIZE, NX, device=DEVICE)
    actor.zero_grad()
    X_opt, U_opt = actor.forward(test_input)
    dummy_loss = U_opt.sum()
    dummy_loss.backward()
    grad = actor.cost_map_net[4].weight.grad

    print("\n--- Risultato del Test ---\n")
    if grad is not None and torch.abs(grad).sum() > 0:
        print("✅ SUCCESSO: Il gradiente per `cost_map_net` è stato calcolato correttamente.")
        print(f"   Somma assoluta del gradiente: {torch.abs(grad).sum().item():.4e}")
        print("   Questo dimostra che la nuova `ILQRSolve` propaga i gradienti alla rete neurale.")
    elif grad is not None:
        print("❌ FALLIMENTO PARZIALE: Il gradiente è un tensore di zeri.")
    else:
        print("❌ FALLIMENTO: Il gradiente è `None`. Il flusso è interrotto.")
except Exception as e:
    print(f"\nSi è verificato un errore durante l'esecuzione del test: {e}")
    import traceback
    traceback.print_exc()
