
# File: checkpoint_manager.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp.grad_scaler import GradScaler
from pathlib import Path
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class CheckpointManager:
    """
    Gestisce il salvataggio e il caricamento dei checkpoint di addestramento.

    Questa classe si occupa di:
    1. Salvare lo stato completo del training, inclusi i modelli (attore e critico),
       gli ottimizzatori e lo scaler per la precisione mista (AMP).
    2. Trovare l'ultimo checkpoint disponibile in una directory.
    3. Caricare lo stato del training per permettere una ripresa senza interruzioni.
    """

    def __init__(self, checkpoint_dir: str):
        """
        Inizializza il gestore di checkpoint.

        Args:
            checkpoint_dir (str): La directory dove salvare e da cui caricare i checkpoint.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        # Crea la directory se non esiste
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_loss = float('inf')

    def save(self, step: int, actor: nn.Module, critic: nn.Module,
             actor_optimizer: optim.Optimizer, critic_optimizer: optim.Optimizer,
             scaler: GradScaler, current_loss: Optional[float] = None):
        """
        Salva lo stato corrente del training in un file .pt.

        Args:
            step (int): Lo step di addestramento corrente.
            actor (nn.Module): Il modello dell'attore.
            critic (nn.Module): Il modello del critico.
            actor_optimizer (optim.Optimizer): L'ottimizzatore dell'attore.
            critic_optimizer (optim.Optimizer): L'ottimizzatore del critico.
            scaler (GradScaler): Lo scaler AMP.
            current_loss (Optional[float]): La loss corrente per salvare il modello "migliore".
        """
        checkpoint_state = {
            'step': step,
            'actor_state_dict': actor.state_dict(),
            'critic_state_dict': critic.state_dict(),
            'actor_optimizer_state_dict': actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': critic_optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_loss': self.best_loss
        }

        # Salva il checkpoint regolare
        filename = self.checkpoint_dir / f"checkpoint_step_{step}.pt"
        torch.save(checkpoint_state, filename)
        logging.info(f"Checkpoint salvato in: {filename}")

        # Salva anche come ultimo checkpoint per un facile accesso
        latest_filename = self.checkpoint_dir / "checkpoint_latest.pt"
        torch.save(checkpoint_state, latest_filename)

        # Salva il checkpoint migliore se la loss è migliorata
        if current_loss is not None and current_loss < self.best_loss:
            self.best_loss = current_loss
            checkpoint_state['best_loss'] = self.best_loss
            best_filename = self.checkpoint_dir / "checkpoint_best.pt"
            torch.save(checkpoint_state, best_filename)
            logging.info(f"Nuovo checkpoint migliore salvato con loss: {current_loss:.4f}")

    def load(self, actor: nn.Module, critic: nn.Module,
             actor_optimizer: optim.Optimizer, critic_optimizer: optim.Optimizer,
             scaler: GradScaler, device: torch.device, resume_from: str = "latest"):
        """
        Carica l'ultimo checkpoint disponibile se esiste.

        Args:
            actor (nn.Module): Il modello dell'attore da inizializzare.
            critic (nn.Module): Il modello del critico da inizializzare.
            actor_optimizer (optim.Optimizer): L'ottimizzatore dell'attore.
            critic_optimizer (optim.Optimizer): L'ottimizzatore del critico.
            scaler (GradScaler): Lo scaler AMP.
            device (torch.device): Il device su cui mappare il checkpoint.
            resume_from (str): "latest" o "best" per scegliere quale checkpoint caricare.

        Returns:
            int: Lo step da cui riprendere l'addestramento (0 se nessun checkpoint trovato).
        """
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{resume_from}.pt"

        if not checkpoint_path.exists():
            logging.warning(f"Nessun checkpoint '{resume_from}' trovato in {self.checkpoint_dir}. Inizio da zero.")
            return 0

        logging.info(f"Caricamento del checkpoint da: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        actor.load_state_dict(checkpoint['actor_state_dict'])
        critic.load_state_dict(checkpoint['critic_state_dict'])
        actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

        if 'best_loss' in checkpoint:
            self.best_loss = checkpoint['best_loss']

        start_step = checkpoint['step'] + 1
        logging.info(f"Training ripreso con successo dallo step {start_step}")

        return start_step
