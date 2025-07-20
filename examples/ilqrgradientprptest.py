# test_nuovo_codice.py (Test corretto per la nuova implementazione)

import torch
import numpy as np

# Assicurati che i file siano nella stessa directory o nel python path
# Potrebbe essere necessario aggiustare gli import in base alla struttura del tuo progetto
from DifferentialMPC import DifferentiableMPCController, GradMethod
from DifferentialMPC import GeneralQuadCost


# --- Definizione di un ActorMPC fittizio per il test ---
# Poiché non hai fornito ActorMPC, ne creo una versione minima
# che riproduce la logica essenziale: una rete che genera parametri di costo
# e un controller MPC che li usa.

class MockActorMPC(torch.nn.Module):
    def __init__(self, nx, nu, horizon, dt, f_dyn, device):
        super().__init__()
        self.device = device

        # Rete Neurale che mappa lo stato ai parametri di costo
        # (C, c, C_final, c_final)
        self.n_tau = nx + nu
        T = horizon
        cost_out_dim = (T * self.n_tau * self.n_tau) + (T * self.n_tau) + \
                       (self.n_tau * self.n_tau) + self.n_tau

        self.cost_map_net = torch.nn.Sequential(
            torch.nn.Linear(nx, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, cost_out_dim)
        ).to(device)

        # Il modulo di costo iniziale è un placeholder. Sarà aggiornato dinamicamente.
        C_init = torch.zeros(T, self.n_tau, self.n_tau, device=device)
        c_init = torch.zeros(T, self.n_tau, device=device)
        C_final_init = torch.zeros(self.n_tau, self.n_tau, device=device)
        c_final_init = torch.zeros(self.n_tau, device=device)

        cost_module = GeneralQuadCost(
            nx, nu, C=C_init, c=c_init, C_final=C_final_init, c_final=c_final_init,
            device=device
        )

        # Controller MPC
        self.mpc_controller = DifferentiableMPCController(
            f_dyn=f_dyn,
            total_time=T * dt,
            step_size=dt,
            horizon=horizon,
            cost_module=cost_module,
            device=device,
            grad_method=GradMethod.ANALYTIC,  # Uso analytic perché forniamo la jacobiana
            f_dyn_jac=f_dyn_jac_batched
        )

    def forward(self, x0_batch):
        B = x0_batch.shape[0]
        T, nx, nu, n_tau = self.mpc_controller.horizon, self.mpc_controller.nx, self.mpc_controller.nu, self.n_tau

        # 1. La rete produce un vettore di parametri di costo
        cost_params = self.cost_map_net(x0_batch)

        # 2. Riorganizza il vettore nei tensori C, c, C_final, c_final
        C_flat, c_flat, C_final_flat, c_final_flat = torch.split(cost_params, [
            T * n_tau * n_tau,
            T * n_tau,
            n_tau * n_tau,
            n_tau
        ], dim=-1)

        C = C_flat.reshape(B, T, n_tau, n_tau)
        c = c_flat.reshape(B, T, n_tau)
        C_final = C_final_flat.reshape(B, n_tau, n_tau)
        c_final = c_final_flat.reshape(B, n_tau)

        # Aggiorna il modulo costo del controller
        # N.B. La nuova `ILQRSolve` prende C e c come input diretti, quindi questo
        # passaggio è per la coerenza del forward pass nel `solve_step`
        self.mpc_controller.cost_module.C = C
        self.mpc_controller.cost_module.c = c
        self.mpc_controller.cost_module.C_final = C_final
        self.mpc_controller.cost_module.c_final = c_final

        # 3. Chiama il forward del controller
        # La nuova `forward` chiama `ILQRSolve.apply` internamente
        X_opt, U_opt = self.mpc_controller.forward(x0_batch)

        return X_opt, U_opt


print("=" * 60)
print("🧪 ESECUZIONE TEST DI PROPAGAZIONE DEL GRADIENTE (CORRETTO) 🧪")
print("Sulla tua nuova implementazione...")
print("=" * 60)

# --- Parametri di base per il test ---
NX, NU, HORIZON, DT = 2, 1, 5, 0.1
DEVICE = "cpu"
BATCH_SIZE = 4


# --- Dinamica e Jacobiana fittizie per l'inizializzazione ---
def f_dyn_batched(x, u, dt=DT):
    p, v = torch.split(x, 1, dim=-1)
    u = u.to(p)
    p_new, v_new = p + v * dt, v + u * dt
    return torch.cat([p_new, v_new], dim=-1)


def f_dyn_jac_batched(x, u, dt=DT):
    B = x.shape[0]
    A = torch.tensor([[1, dt], [0, 1]], device=x.device, dtype=x.dtype).unsqueeze(0).expand(B, -1, -1)
    B_mat = torch.tensor([[0], [dt]], device=x.device, dtype=x.dtype).unsqueeze(0).expand(B, -1, -1)
    return A, B_mat


# --- Esecuzione del Test ---
try:
    actor = MockActorMPC(
        nx=NX, nu=NU, horizon=HORIZON, dt=DT,
        f_dyn=f_dyn_batched, device=DEVICE
    )

    test_input = torch.randn(BATCH_SIZE, NX, device=DEVICE)
    actor.zero_grad()

    X_opt, U_opt = actor.forward(test_input)

    # Calcola una loss fittizia sull'output
    dummy_loss = U_opt.sum()

    # Esegui il backward pass
    dummy_loss.backward()

    # Controlla il gradiente di un layer specifico della rete
    grad = actor.cost_map_net[4].weight.grad

    print("\n--- Risultato del Test ---\n")
    if grad is not None and torch.abs(grad).sum() > 0:
        print("✅ SUCCESSO: Il gradiente per `cost_map_net` è stato calcolato correttamente.")
        print(f"   Somma assoluta del gradiente: {torch.abs(grad).sum().item():.4e}")
        print("   Questo dimostra che la nuova `ILQRSolve` propaga i gradienti alla rete neurale.")
    elif grad is not None:
        print("❌ FALLIMENTO PARZIALE: Il gradiente è un tensore di zeri.")
        print("   La propagazione avviene, ma il risultato è nullo. Controllare la loss o la catena di calcoli.")
    else:
        print("❌ FALLIMENTO: Il gradiente è ancora `None`.")
        print("   C'è un problema residuo che blocca il flusso dei gradienti.")

except Exception as e:
    print(f"\nSi è verificato un errore durante l'esecuzione del test: {e}")
    import traceback

    traceback.print_exc()