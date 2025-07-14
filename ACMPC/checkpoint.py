
import torch
import os
def save_checkpoint(actor, critic_1, critic_2, actor_optimizer, critic_optimizer, episode, filename="checkpoint.pth"):
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
