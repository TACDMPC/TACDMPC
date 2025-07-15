import torch
from torch import nn, Tensor
from transformers import BertConfig, BertModel

class CriticTransformer(nn.Module):
    """Critic basato su un piccolo Transformer HuggingFace."""

    def __init__(
        self,
        nx: int,
        nu: int,
        history_len: int,
        horizon: int,
        hidden_size: int = 64,
        n_layers: int = 2,
        n_heads: int = 2,
    ) -> None:
        super().__init__()
        self.nx = nx
        self.nu = nu
        self.history_len = history_len
        self.horizon = horizon
        config = BertConfig(
            vocab_size=1,
            hidden_size=hidden_size,
            num_hidden_layers=n_layers,
            num_attention_heads=n_heads,
            intermediate_size=hidden_size * 4,
            max_position_embeddings=history_len + horizon + 1,
        )
        self.model = BertModel(config)
        self.input_proj = nn.Linear(nx + nu, hidden_size)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(
        self,
        x_t: Tensor,
        u_t: Tensor,
        x_past: Tensor,
        u_past: Tensor,
        x_pred: Tensor,
        u_pred: Tensor,
    ) -> Tensor:
        tokens = torch.cat([
            torch.cat([x_past, u_past], dim=-1),
            torch.cat([x_t.unsqueeze(1), u_t.unsqueeze(1)], dim=-1),
            torch.cat([x_pred, u_pred], dim=-1),
        ], dim=1)
        emb = self.input_proj(tokens)
        out = self.model(inputs_embeds=emb).last_hidden_state
        pooled = out[:, -1, :]
        return self.value_head(pooled).squeeze(-1)
