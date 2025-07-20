# gradientflowtest.py (Versione di DEBUG con loss diretta)

import torch
import torch.nn as nn
import sys

# --- Importazioni delle tue classi dai file locali ---
try:
    from ACMPC import ActorMPC
    from ACMPC import CriticTransformer
    print("✅ Classi 'ActorMPC' e 'CriticTransformer' importate correttamente.")
except ImportError as e:
    print(f"❌ Errore di importazione: {e}")
    print("Assicurati che i file 'actor.py' e 'critic_transformer.py' e 'DifferentialMPC.py' siano nella stessa directory.")
    sys.exit()

# --- Funzione helper per controllare e stampare lo stato dei gradienti ---
def check_gradients(model_name, parameter_name, param):
    """Controlla e stampa lo stato del gradiente per un dato parametro."""
    print(f"- Controllando: {model_name} -> {parameter_name}")
    grad = param.grad
    if grad is None:
        print("  ❌ RISULTATO: Gradiente è None (Nessun flusso di gradiente).")
        return False
    elif torch.abs(grad).sum() == 0:
        print("  ⚠️ RISULTATO: Gradiente è un tensore di ZERI (Il flusso esiste, ma la derivata è nulla).")
        return False
    else:
        print("  ✅ RISULTATO: Gradiente VALIDO (Flusso di gradiente corretto).")
        return True

# --- Setup del Test ---
print("\n" + "="*70)
print(" 🧪 ESECUZIONE TEST DI DEBUG CON LOSS DIRETTA 🧪 ")
print("Verificheremo il flusso dei gradienti usando una loss semplificata.")
print("="*70)

# Parametri di base
NX, NU, HORIZON, DT = 2, 1, 5, 0.1
DEVICE = "cpu"
BATCH_SIZE = 4
HISTORY_LEN = 10

# Dinamica fittizia per inizializzazione
def f_dyn_batched(x, u, dt=DT):
    p, v = torch.split(x, 1, dim=-1)
    u = u.to(p)
    p_new, v_new = p + v * dt, v + u * dt
    return torch.cat([p_new, v_new], dim=-1)

# Inizializzazione dei modelli
try:
    # Assumiamo che actor.py contenga le ultime modifiche (reg_eps, ecc.)
    actor = ActorMPC(nx=NX, nu=NU, horizon=HORIZON, dt=DT, f_dyn=f_dyn_batched, device=DEVICE)
    critic = CriticTransformer(state_dim=NX, action_dim=NU, history_len=HISTORY_LEN, pred_horizon=HORIZON)
except Exception as e:
    print(f"❌ Errore durante l'inizializzazione dei modelli: {e}")
    sys.exit()

# Azzera tutti i gradienti
actor.zero_grad()
critic.zero_grad()

# --- Forward Pass ---
current_states = torch.randn(BATCH_SIZE, NX, device=DEVICE)
history_tokens = torch.randn(BATCH_SIZE, HISTORY_LEN, NX + NU)

# Eseguiamo il forward dell'attore
action, log_prob, predicted_states, predicted_actions = actor.forward(current_states)
# Eseguiamo il forward del critico
values = critic(history_tokens)

# --- Calcolo delle Loss (MODIFICATO PER IL DEBUG) ---
print("\nATTENZIONE: Verrà usata una loss di test diretta (predicted_actions.sum()) per il debug.\n")

# --- MODIFICA CHIAVE ---
# Bypassiamo la loss basata su log_prob e usiamo una loss diretta sull'output del MPC
# per verificare che il solver stesso sia differenziabile.
actor_loss_debug = predicted_actions.sum()

# La loss del critico rimane invariata ma la calcoliamo per completezza
dummy_returns = torch.randn_like(values)
critic_loss = nn.functional.mse_loss(values, dummy_returns)

# Usiamo solo la loss dell'attore per il backward pass per isolare il problema
total_loss = actor_loss_debug + critic_loss

# --- Backward Pass ---
print("Eseguo il backward pass sulla loss totale...\n")
total_loss.backward()

# --- Ispezione dei Gradienti ---
print("--- 1. Analisi Gradienti ATTORE (ActorMPC) ---")
all_ok = check_gradients("Attore", "cost_map_net (ultimo layer)", actor.cost_map_net[4].weight)
all_ok &= check_gradients("Attore", "log_std", actor.log_std)

print("\n--- 2. Analisi Gradienti CRITICO (CriticTransformer) ---")
all_ok &= check_gradients("Critico", "Embedding Layer", critic.embed.weight)
all_ok &= check_gradients("Critico", "Output Head Layer", critic.head.weight)

print("\n" + "="*70)
print("Conclusione del Test")
print("="*70)

if all_ok:
    print("🎉 SUCCESSO DEL DEBUG! 🎉")
    print("Il test con loss diretta ha prodotto gradienti VALIDI per la `cost_map_net`.")
    print("Questo conferma che il tuo solver MPC è corretto e differenziabile.")
    print("\nPROSSIMI PASSI:")
    print("Il problema del gradiente nullo nel tuo training loop è definitivamente")
    print("isolato nel calcolo della loss basata su `log_prob`. Concentra il debug lì.")
else:
    print("PROBLEMA RILEVATO.")
    print("Anche con la loss diretta, uno o più componenti non ricevono gradienti.")
    print("Controllare i risultati sopra.")