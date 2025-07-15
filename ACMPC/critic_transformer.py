import torch
import torch.nn as nn
from transformers import BertConfig, BertModel


class CriticTransformer(nn.Module):
    """Transformer critic built on Hugging Face BertModel."""

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
        config = BertConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=4 * hidden_size,
            vocab_size=1,
            max_position_embeddings=history_len + pred_horizon + 1,
        )
        self.transformer = BertModel(config)
        self.head = nn.Linear(hidden_size, 1)

    def forward(
        self,
        current_state: torch.Tensor,
        current_action: torch.Tensor,
        history: torch.Tensor,
        predicted: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return Q-value for given trajectory pieces."""
        if predicted is None:
            predicted = torch.zeros(
                current_state.shape[0],
                self.pred_horizon,
                self.token_dim,
                device=current_state.device,
                dtype=current_state.dtype,
            )
        seq = torch.cat(
            [history, torch.cat([current_state, current_action], dim=-1).unsqueeze(1), predicted],
            dim=1,
        )
        x = self.embed(seq)
        outputs = self.transformer(inputs_embeds=x)
        last = outputs.last_hidden_state[:, -1, :]
        q = self.head(last)
        return q.squeeze(-1)
