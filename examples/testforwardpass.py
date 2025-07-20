# testforwardpass.py (Versione Definitiva e Corretta)

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- Importazioni dirette dai tuoi file ---
try:
    from DifferentialMPC import DifferentiableMPCController, GeneralQuadCost
except ImportError as e:
    print(f"Errore di importazione: {e}")
    print("Assicurati che questo script sia nella stessa cartella dei file del progetto.")
    exit()

print("=" * 60)
print("🧪 ESECUZIONE TEST DEL FORWARD PASS DEL CONTROLLER MPC 🧪")
print("=" * 60)

# --- Parametri di base per il test ---
NX, NU, HORIZON, DT = 2, 1, 20, 0.1
DEVICE = "cpu"


# --- Dinamica per il test ---
def f_dyn_batched(x, u, dt=DT):
    p, v = torch.split(x, 1, dim=-1)
    u = u.to(p)
    p_new, v_new = p + v * dt, v + u * dt
    return torch.cat([p_new, v_new], dim=-1)


# --- Test ---
try:
    print("1. Configurazione del problema di stabilizzazione...")
    Q = torch.diag(torch.tensor([10.0, 1.0], device=DEVICE))
    R = torch.eye(NU, device=DEVICE) * 0.1
    C_running = torch.block_diag(Q, R).unsqueeze(0).expand(HORIZON, -1, -1)
    c_running = torch.zeros(HORIZON, NX + NU, device=DEVICE)
    C_final = torch.block_diag(Q, torch.zeros(NU, NU, device=DEVICE))
    c_final = torch.zeros(NX + NU, device=DEVICE)

    cost_module = GeneralQuadCost(
        nx=NX, nu=NU, C=C_running, c=c_running,
        C_final=C_final, c_final=c_final, device=DEVICE
    )

    mpc = DifferentiableMPCController(
        f_dyn=f_dyn_batched, total_time=HORIZON * DT, step_size=DT,
        horizon=HORIZON, cost_module=cost_module, device=DEVICE
    )

    print("2. Esecuzione del solver MPC da uno stato iniziale [3.0, 0.0]...")
    x0 = torch.tensor([[3.0, 0.0]], device=DEVICE)
    U_init = torch.zeros(1, HORIZON, NU, device=DEVICE)

    with torch.no_grad():
        # --- MODIFICA CHIAVE: Recuperiamo i risultati dal return della funzione ---
        x_pred, u_mpc = mpc.solve_step(x0, U_init)

    # Converti i risultati in numpy per l'analisi e il plot
    x_pred_np = x_pred.cpu().numpy().squeeze()
    u_mpc_np = u_mpc.cpu().numpy().squeeze()

    print("3. Analisi dei risultati...")
    final_state = x_pred_np[-1]
    is_stable = np.all(np.abs(final_state) < 0.1)

    if not np.any(np.isnan(x_pred_np)) and not np.any(np.isnan(u_mpc_np)):
        print("✅ SUCCESSO: Il solver non ha prodotto NaN.")
    else:
        print("❌ FALLIMENTO: Il solver ha prodotto valori NaN.")
        exit()

    if is_stable:
        print(f"✅ SUCCESSO: Lo stato finale {final_state} è vicino all'origine (0,0).")
        print("Il controller ha risolto correttamente il problema di stabilizzazione.")
    else:
        print(f"⚠️  ATTENZIONE: Lo stato finale {final_state} non è vicino all'origine.")
        print("Il forward pass converge, ma potrebbe non essere ottimale.")

    print("\n4. Generazione del grafico della traiettoria...")
    Path("test_plots").mkdir(exist_ok=True)
    time = np.arange(x_pred_np.shape[0]) * DT

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle('Validazione Forward Pass MPC', fontsize=16)
    axs[0].plot(time, x_pred_np[:, 0], 'b-o', markersize=3, label='Posizione (p)')
    axs[0].set_ylabel('Posizione');
    axs[0].grid(True, ls=':');
    axs[0].legend()
    axs[1].plot(time, x_pred_np[:, 1], 'g-o', markersize=3, label='Velocità (v)')
    axs[1].set_ylabel('Velocità');
    axs[1].grid(True, ls=':');
    axs[1].legend()
    axs[2].plot(time[:-1], u_mpc_np, 'r', drawstyle='steps-post', label='Controllo (a)')
    axs[2].set_ylabel('Controllo');
    axs[2].set_xlabel('Tempo (s)');
    axs[2].grid(True, ls=':');
    axs[2].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path = "test_plots/forward_pass_trajectory.png"
    plt.savefig(plot_path)
    print(f"Grafico salvato in: {plot_path}")
    plt.close(fig)

except Exception as e:
    import traceback

    print(f"\nSi è verificato un errore imprevisto durante il test: {e}")
    traceback.print_exc()