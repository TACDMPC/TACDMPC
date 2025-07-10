
import torch
import os


def save_checkpoint(actor, critic_1, critic_2, actor_optimizer, critic_optimizer, episode, filename="checkpoint.pth"):
    """
    Salva lo stato corrente del training in un file.

    Args:
        actor (nn.Module): Il modello dell'attore.
        critic_1 (nn.Module): Il primo modello del critico.
        critic_2 (nn.Module): Il secondo modello del critico.
        actor_optimizer (Optimizer): L'ottimizzatore dell'attore.
        critic_optimizer (Optimizer): L'ottimizzatore dei critici.
        episode (int): Il numero dell'episodio corrente.
        filename (str): Il nome del file in cui salvare il checkpoint.
    """
    print(f"💾 Salvataggio checkpoint su '{filename}'...")
    state = {
        'episode': episode,
        'actor_state_dict': actor.state_dict(),
        'critic_1_state_dict': critic_1.state_dict(),
        'critic_2_state_dict': critic_2.state_dict(),
        'actor_optimizer_state_dict': actor_optimizer.state_dict(),
        'critic_optimizer_state_dict': critic_optimizer.state_dict(),
    }
    torch.save(state, filename)
    print("   Checkpoint salvato con successo.")


def load_checkpoint(actor, critic_1, critic_2, actor_optimizer, critic_optimizer, filename="checkpoint.pth",
                    device="cpu"):
    """
    Carica lo stato del training da un file di checkpoint.

    Args:
        actor (nn.Module): Il modello dell'attore da popolare.
        critic_1 (nn.Module): Il primo critico da popolare.
        critic_2 (nn.Module): Il secondo critico da popolare.
        actor_optimizer (Optimizer): L'ottimizzatore dell'attore da popolare.
        critic_optimizer (Optimizer): L'ottimizzatore dei critici da popolare.
        filename (str): Il nome del file da cui caricare il checkpoint.
        device (str): Il dispositivo su cui mappare il checkpoint.

    Returns:
        int: Il numero dell'episodio da cui riprendere l'addestramento.
             Restituisce 0 se non viene trovato alcun checkpoint.
    """
    if not os.path.isfile(filename):
        print(f"⚠️ Nessun checkpoint trovato in '{filename}'. Inizio un nuovo addestramento.")
        return 0

    print(f"📂 Caricamento checkpoint da '{filename}'...")
    checkpoint = torch.load(filename, map_location=device)

    actor.load_state_dict(checkpoint['actor_state_dict'])
    critic_1.load_state_dict(checkpoint['critic_1_state_dict'])
    critic_2.load_state_dict(checkpoint['critic_2_state_dict'])
    actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
    critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

    start_episode = checkpoint['episode'] + 1

    print(f"   Checkpoint caricato. Si riprenderà dall'episodio {start_episode}.")

    return start_episode


# --- Esempio di utilizzo (Sanity Check) ---
if __name__ == '__main__':
    import torch.nn as nn
    import torch.optim as optim

    # Creiamo modelli e ottimizzatori fittizi
    dummy_actor = nn.Linear(10, 2)
    dummy_critic1 = nn.Linear(12, 1)
    dummy_critic2 = nn.Linear(12, 1)

    dummy_actor_opt = optim.Adam(dummy_actor.parameters())
    dummy_critic_opt = optim.Adam(list(dummy_critic1.parameters()) + list(dummy_critic2.parameters()))

    # 1. SALVATAGGIO
    current_episode = 50
    checkpoint_file = "test_checkpoint.pth"
    save_checkpoint(dummy_actor, dummy_critic1, dummy_critic2, dummy_actor_opt, dummy_critic_opt, current_episode,
                    checkpoint_file)

    # 2. CARICAMENTO
    # Creiamo nuove istanze vuote
    new_actor = nn.Linear(10, 2)
    new_critic1 = nn.Linear(12, 1)
    new_critic2 = nn.Linear(12, 1)

    new_actor_opt = optim.Adam(new_actor.parameters())
    new_critic_opt = optim.Adam(list(new_critic1.parameters()) + list(new_critic2.parameters()))

    # Verifichiamo che i pesi siano diversi prima del caricamento
    assert not torch.equal(dummy_actor.weight, new_actor.weight)

    # Carichiamo lo stato salvato
    resume_episode = load_checkpoint(new_actor, new_critic1, new_critic2, new_actor_opt, new_critic_opt,
                                     checkpoint_file)

    # Verifichiamo che i pesi ora siano identici e l'episodio sia corretto
    assert torch.equal(dummy_actor.weight, new_actor.weight)
    assert resume_episode == current_episode + 1

    print("\n✅ Sanity check per il checkpoint superato con successo.")

    # Pulizia del file di test
    os.remove(checkpoint_file)