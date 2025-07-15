from __future__ import annotations
import torch
from torch import nn
from transformers import BertConfig, BertModel


class CriticTransformer(nn.Module):
    """Transformer critic implemented with HuggingFace Bert layers."""

    def __init__(
        self,
        nx: int,
        nu: int,
        history_len: int,
        *,
        hidden_size: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
    ) -> None:
        super().__init__()
        self.history_len = history_len
        input_dim = nx + nu
        config = BertConfig(
            vocab_size=1,
            hidden_size=hidden_size,
            num_hidden_layers=n_layers,
            num_attention_heads=n_heads,
            intermediate_size=hidden_size * 4,
            max_position_embeddings=history_len,
            pad_token_id=0,
        )
        self.transformer = BertModel(config)
        self.input_proj = nn.Linear(input_dim, hidden_size)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        # history shape (B, T, nx+nu)
        emb = self.input_proj(history)
        out = self.transformer(inputs_embeds=emb).last_hidden_state
        last = out[:, -1, :]
        return self.output_head(last).squeeze(-1)
