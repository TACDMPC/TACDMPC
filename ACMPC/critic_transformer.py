import torch
import torch.nn as nn
from transformers import BertConfig, BertModel
from typing import Optional


class CriticTransformer(nn.Module):
    """
    Transformer critic built on Hugging Face BertModel.
    accetta un tensore di token
    e opzionalmente una sequenza di token predetti.
    """

    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            history_len: int,
            pred_horizon: int,
            hidden_size: int = 64,
            num_layers: int = 2,
            num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.history_len = history_len
        self.pred_horizon = pred_horizon
        self.token_dim = state_dim + action_dim
        self.embed = nn.Linear(self.token_dim, hidden_size)

        # La massima lunghezza della sequenza deve considerare la storia e le predizioni
        max_seq_length = history_len + pred_horizon + 1

        config = BertConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=4 * hidden_size,
            vocab_size=1,  # Non usato, ma richiesto da BertConfig
            max_position_embeddings=max_seq_length,
        )
        self.transformer = BertModel(config)
        self.head = nn.Linear(hidden_size, 1)

    def forward(
            self,
            history_tokens: torch.Tensor,
            predicted_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Restituisce il Q-valore per le sequenze di input.
        Args:
            history_tokens (torch.Tensor): Tensore di token passati (storia).
                                           Shape: (B, history_len, token_dim)
            predicted_tokens (Optional[torch.Tensor]): Tensore opzionale di token futuri predetti.
                                                       Shape: (B, pred_horizon, token_dim)
        Returns:
            torch.Tensor: Q-valore stimato. Shape: (B, 1)
        """
        # Se i token predetti sono forniti, li concateniamo alla storia.
        # Altrimenti, usiamo solo la storia.
        if predicted_tokens is not None:
            # Concatena la storia con le predizioni lungo la dimensione della sequenza
            seq = torch.cat([history_tokens, predicted_tokens], dim=1)
        else:
            seq = history_tokens
        #Embedding della sequenza di input
        x = self.embed(seq)
        # Passaggio attraverso il Transformer
        # Non è necessario un attention_mask se tutte le sequenze hanno la stessa lunghezza (secondo chatgpt devo verificare)
        outputs = self.transformer(inputs_embeds=x)
        # Estrazione dell'output dell'ultimo token
        # L'output 'last_hidden_state' ha shape (B, seq_len, hidden_size)
        # Prendiamo l'output corrispondente all'ultimo token della sequenza.
        last_token_output = outputs.last_hidden_state[:, -1, :]
        # Q-v finale
        q = self.head(last_token_output)
        # Rimuove l'ultima dimensione per avere un output di shape (B,)
        return q.squeeze(-1)