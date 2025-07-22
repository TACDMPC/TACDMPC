import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp.grad_scaler import GradScaler
from pathlib import Path
import logging
import glob
from typing import Optional, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class CheckpointManager:
    """
    Gestisce il salvataggio e il caricamento dei checkpoint di addestramento.
    """

    def __init__(self, checkpoint_dir: str, max_to_keep: int = 5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_to_keep = max_to_keep
        self.best_loss = float('inf')

    def save(self,
             # --- MODIFICA 1: Rinominiamo 'step' in 'completed_step_idx' per chiarezza ---
             completed_step_idx: int,
             actor: nn.Module,
             critic: nn.Module,
             actor_optimizer: optim.Optimizer,
             critic_optimizer: optim.Optimizer,
             scaler: GradScaler,
             current_loss: float,
             metadata: Optional[Dict[str, Any]] = None):
        is_best = current_loss < self.best_loss
        if is_best:
            self.best_loss = current_loss

        checkpoint_state = {
            # --- MODIFICA 2: Salva l'indice dello step appena completato ---
            'completed_step_idx': completed_step_idx,
            'actor_state_dict': actor.state_dict(),
            'critic_state_dict': critic.state_dict(),
            'actor_optimizer_state_dict': actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': critic_optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_loss': self.best_loss,
            'metadata': metadata if metadata is not None else {}
        }

        # Usa l'indice (step number = index + 1) per il nome del file per coerenza
        filename = self.checkpoint_dir / f"checkpoint_step_{completed_step_idx + 1}.pt"
        torch.save(checkpoint_state, filename)
        logging.info(f"Checkpoint salvato in: {filename}")

        self._cleanup_checkpoints()

        latest_filename = self.checkpoint_dir / "checkpoint_latest.pt"
        torch.save(checkpoint_state, latest_filename)

        if is_best:
            best_filename = self.checkpoint_dir / "checkpoint_best.pt"
            torch.save(checkpoint_state, best_filename)
            logging.info(f"✨ Nuovo checkpoint migliore salvato con loss: {current_loss:.4f}")

    def load(self,
             device: torch.device,
             resume_from: str = "latest",
             actor: Optional[nn.Module] = None,
             critic: Optional[nn.Module] = None,
             actor_optimizer: Optional[optim.Optimizer] = None,
             critic_optimizer: Optional[optim.Optimizer] = None,
             scaler: Optional[GradScaler] = None) -> int:
        """
        Carica un checkpoint in modo flessibile.
        Restituisce l'INDICE del prossimo step da eseguire.
        """
        if Path(resume_from).is_file():
            checkpoint_path = Path(resume_from)
        else:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_{resume_from}.pt"

        if not checkpoint_path.exists():
            logging.warning(f"⚠️ Nessun checkpoint trovato in '{checkpoint_path}'. Inizio dall'indice 0.")
            return 0

        logging.info(f"Caricamento del checkpoint da: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        if actor and 'actor_state_dict' in checkpoint:
            actor.load_state_dict(checkpoint['actor_state_dict'])
        if critic and 'critic_state_dict' in checkpoint:
            critic.load_state_dict(checkpoint['critic_state_dict'])
        if actor_optimizer and 'actor_optimizer_state_dict' in checkpoint:
            actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        if critic_optimizer and 'critic_optimizer_state_dict' in checkpoint:
            critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        if scaler and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.best_loss = checkpoint.get('best_loss', float('inf'))

        # --- MODIFICA 3: Logica di ripresa basata sull'indice completato ---
        completed_idx = checkpoint.get('completed_step_idx', -1)
        start_idx = completed_idx + 1

        logging.info(
            f"✅ Training ripreso. Ultimo step completato (indice): {completed_idx}. Si riparte dall'indice: {start_idx}.")
        return start_idx

    def _cleanup_checkpoints(self):
        if self.max_to_keep <= 0: return
        checkpoints = sorted(
            glob.glob(str(self.checkpoint_dir / "checkpoint_step_*.pt")),
            key=lambda x: int(Path(x).stem.split('_')[-1])
        )
        if len(checkpoints) > self.max_to_keep:
            for ckpt_path in checkpoints[:-self.max_to_keep]:
                Path(ckpt_path).unlink()
                logging.debug(f"Rimosso vecchio checkpoint: {ckpt_path}")