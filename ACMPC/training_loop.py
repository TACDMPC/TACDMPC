# File: training_loop.py (Versione Modificata)

import torch
import torch.optim as optim
from contextlib import nullcontext
from importlib import util as import_util
from pathlib import Path
from typing import Callable  # MODIFICA: Aggiunto Callable per type hinting
from utils.profiler import Profiler

# Importazione di utils come nel codice originale
try:
    from utils import seed_everything
except Exception:
    spec = import_util.spec_from_file_location(
        "utils_module", Path(__file__).resolve().parents[1] / "utils.py"
    )
    _mod = import_util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(_mod)
    seed_everything = _mod.seed_everything

from .actor import ActorMPC
from .critic_transformer import CriticTransformer

# La funzione rollout rimane invariata
def rollout(env, actor: ActorMPC, mpc_horizon: int):
    """Esegue un episodio e raccoglie anche le predizioni dell'MPC per MPVE."""
    states, actions, rewards, log_probs = [], [], [], []
    # Liste per le predizioni
    predicted_states_list, predicted_actions_list = [], []

    state = env.reset()

    for _ in range(env.sim_horizon):
        # La chiamata all'attore ora restituisce più valori
        action, log_prob, pred_states, pred_actions = actor(state)

        new_state, reward, done = env.step(action)

        states.append(state.cpu())
        actions.append(action.cpu())
        rewards.append(torch.tensor([reward]))
        log_probs.append(log_prob.cpu())

        # Colleziona le predizioni
        predicted_states_list.append(pred_states.cpu())
        predicted_actions_list.append(pred_actions.cpu())

        state = new_state
        if done:
            break

    return (
        torch.stack(states),
        torch.stack(actions),
        torch.stack(rewards),
        torch.stack(log_probs),
        torch.stack(predicted_states_list),
        torch.stack(predicted_actions_list)
    )


# La funzione compute_gae_and_returns rimane invariata
def compute_gae_and_returns(
        rewards: torch.Tensor,
        values: torch.Tensor,
        *,
        gamma: float = 0.99,
        lam: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calcola vantaggi e ritorni usando Generalized Advantage Estimation (GAE)."""
    T = rewards.shape[0]
    values_extended = torch.cat([values, torch.zeros(1, device=values.device, dtype=values.dtype)], dim=0)
    advantages = torch.zeros_like(rewards)
    last_advantage = 0.0

    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values_extended[t + 1] - values_extended[t]
        last_advantage = delta + gamma * lam * last_advantage
        advantages[t] = last_advantage

    returns = advantages + values
    return advantages, returns


def train(
        env,
        actor: ActorMPC,
        critic: CriticTransformer,
        reward_fn: Callable,
        steps: int = 1,
        *,
        mpc_horizon: int,
        use_amp: bool = False,
        profile: bool = False,
        log_file: str | None = None,
        track_gpu: bool = False,
        seed: int | None = None,
        mpve_gamma: float = 0.99,
        mpve_weight: float = 0.1,
):
    if seed is not None:
        seed_everything(seed)

    actor_opt = optim.Adam(actor.parameters(), lr=3e-4)
    critic_opt = optim.Adam(critic.parameters(), lr=3e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Pre-calcola le dimensioni per la preparazione dei token
    history_len = critic.history_len
    token_dim = critic.token_dim
    device = next(critic.parameters()).device
    dtype = next(critic.parameters()).dtype

    for step_idx in range(steps):
        with torch.cuda.amp.autocast(enabled=use_amp):
            # ... (profiler etc. rimangono uguali) ...

            states, actions, rewards, log_probs, pred_states, pred_actions = rollout(env, actor, mpc_horizon)

            # --- MODIFICA: Preparazione dei token e calcolo dei valori per GAE ---
            states_detached = states.detach()
            actions_detached = actions.detach()

            # Crea una sequenza unica di token (stato, azione) per l'episodio
            tokens = torch.cat([states_detached, actions_detached], dim=-1)

            # Aggiunge un padding iniziale di zeri per creare le sequenze di storia
            padded_tokens = torch.cat([torch.zeros(history_len - 1, token_dim, device=device, dtype=dtype), tokens],
                                      dim=0)

            # Usa `unfold` per creare un batch di sequenze storiche con una finestra mobile
            # history_batch[t] conterrà la sequenza di token fino al tempo t
            history_batch = padded_tokens.unfold(dimension=0, size=history_len, step=1).permute(0, 2, 1)

            # Calcola i valori per l'episodio reale in un'unica chiamata batch
            # Usiamo solo la storia, senza token predetti
            values = critic(history_tokens=history_batch, predicted_tokens=None)
            if values.ndim == 1: values = values.unsqueeze(1)

            # Calcolo GAE (invariato)
            advantages, returns = compute_gae_and_returns(rewards, values.detach())

            # --- MODIFICA: Calcolo della Loss MPVE (vettorizzato) ---
            T = states.shape[0]  # Lunghezza dell'episodio

            # Calcola le ricompense predette in batch (this part was correct)
            predicted_rewards = reward_fn(
                pred_states[:, :-1, :],  # Stati da t=0 a H-1
                pred_actions,  # Azioni da t=0 a H-1
                pred_states[:, 1:, :]  # Stati successivi da t=1 a H
            )

            # --- CORREZIONE: Preparazione dei token per il calcolo dei valori ---
            # Crea i token per le traiettorie predette. Shape: (T, H, token_dim)
            # H è l'orizzonte di predizione (mpc_horizon)
            predicted_tokens = torch.cat([pred_states[:, :-1, :], pred_actions], dim=-1)

            # Calcola V(s_H) per ogni predizione (valore terminale)
            with torch.no_grad():
                # Per calcolare V(s_H), concateniamo la storia fino a t con l'INTERA sequenza predetta
                # e il transformer estrarrà il valore dall'ultimo token (s_H).
                # Shape history_batch: (T, history_len, token_dim)
                # Shape predicted_tokens: (T, H, token_dim)
                final_pred_input_seq = torch.cat([history_batch, predicted_tokens], dim=1)

                # Calcola il valore alla fine dell'orizzonte di predizione.
                v_final_pred = critic(history_tokens=final_pred_input_seq, predicted_tokens=None).squeeze()

            # Calcola i target MPVE con il TD(k)-trick (this part was correct)
            mpve_targets = torch.zeros_like(predicted_rewards)
            # Itera su ogni traiettoria predetta nel batch dell'episodio
            for i in range(T):
                next_val = v_final_pred[i]
                # Backward pass per calcolare i target k-step
                for t in reversed(range(mpc_horizon)):
                    target = predicted_rewards[i, t] + mpve_gamma * next_val
                    mpve_targets[i, t] = target
                    # Il valore successivo per il target è il target appena calcolato (TD-trick)
                    # Questo è coerente con la formula (9) del paper.
                    next_val = target

            # --- CORREZIONE: Calcolo dei valori predetti per la loss MPVE ---
            # Dobbiamo calcolare V(s_hat_k) per ogni k nell'orizzonte.
            # Dato che il critico è un transformer, dobbiamo costruire la sequenza di input corretta per ogni step.
            predicted_values_list = []
            for t in range(T):  # Itera su ogni step dell'episodio reale
                history_at_t = history_batch[t].unsqueeze(0)  # Shape: (1, history_len, token_dim)

                for k in range(mpc_horizon):  # Itera su ogni step della predizione
                    # L'input per il critico è la storia fino a t, seguita dalla predizione fino a k.
                    # predicted_tokens[t, :k+1, :] -> prende la sequenza predetta dallo step 0 a k
                    # Shape: (1, k+1, token_dim)
                    current_pred_seq = predicted_tokens[t, :k + 1, :].unsqueeze(0)

                    # Concatena la storia con la porzione di predizione
                    input_for_critic = torch.cat([history_at_t, current_pred_seq], dim=1)

                    # Calcola V(s_hat_k)
                    value = critic(history_tokens=input_for_critic)
                    predicted_values_list.append(value)

            # Concatena tutti i valori calcolati in un unico tensore
            predicted_values = torch.cat(predicted_values_list)

            # Calcola la loss MPVE confrontando i valori predetti con i target calcolati
            # predicted_values ha V(s_hat_0), V(s_hat_1)... per ogni t
            # mpve_targets.view(-1) ha i target corrispondenti
            loss_mpve = torch.nn.functional.mse_loss(predicted_values, mpve_targets.view(-1))

            # --- Calcolo Loss Finali (this part was correct) ---
            actor_loss = -(log_probs * advantages).mean()
            critic_loss_gae = torch.nn.functional.mse_loss(values, returns.squeeze(-1))
            critic_loss = critic_loss_gae + mpve_weight * loss_mpve

        # --- MODIFICA: Backward pass e ottimizzazione (semplificato) ---
        actor_opt.zero_grad(set_to_none=True)
        critic_opt.zero_grad(set_to_none=True)

        # Le due loss sono indipendenti, non serve `retain_graph=True`
        scaler.scale(actor_loss).backward()
        scaler.scale(critic_loss).backward()

        scaler.step(actor_opt)
        scaler.step(critic_opt)
        scaler.update()

        print(
            f"Step {step_idx}: Actor Loss: {actor_loss.item():.4f}, Critic Loss: {critic_loss.item():.4f} "
            f"(GAE: {critic_loss_gae.item():.4f}, MPVE: {loss_mpve.item():.4f})"
        )