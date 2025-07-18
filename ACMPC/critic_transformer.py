# File: critic_transformer.py

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel
from typing import Optional


class CriticTransformer(nn.Module):
    """
    Transformer critic che riceve una sequenza di token già assemblata.
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

        # --- MODIFICA CHIAVE: Assicura che la lunghezza massima sia sufficiente ---
        # La sequenza più lunga che il critico vedrà mai è la storia + le predizioni.
        max_seq_length = history_len + pred_horizon

        config = BertConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=4 * hidden_size,
            vocab_size=1,
            max_position_embeddings=max_seq_length,
            is_decoder=True,
        )
        self.transformer = BertModel(config)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, sequence_tokens: torch.Tensor, use_causal_mask: bool = False) -> torch.Tensor:
        """
        Restituisce un valore V(s) per OGNI token nella sequenza.
        """
        x = self.embed(sequence_tokens)

        # Il modello con is_decoder=True gestisce la causalità internamente
        outputs = self.transformer(inputs_embeds=x)

        all_token_outputs = outputs.last_hidden_state
        values = self.head(all_token_outputs)

        return values.squeeze(-1)