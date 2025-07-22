
import torch
import torch.nn as nn
from transformers import BertConfig, BertModel
from typing import Optional

class CriticTransformer(nn.Module):
    """
    Transformer critic adapted to receive only state tokens, as required by the training loop.
    """
    def __init__(
            self,
            state_dim: int,
            action_dim: int, # Kept for signature consistency but not used for embedding size
            history_len: int,
            pred_horizon: int,
            hidden_size: int = 64,
            num_layers: int = 2,
            num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.token_dim = state_dim
        self.embed = nn.Linear(self.token_dim, hidden_size)
        # The longest sequence the critic sees is the predicted MPC trajectory.
        max_seq_length = pred_horizon + history_len

        config = BertConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=4 * hidden_size,
            vocab_size=1, # Not used when providing embeddings
            max_position_embeddings=max_seq_length,
            is_decoder=True, # Handles causal masking internally
        )
        self.transformer = BertModel(config)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, sequence_tokens: torch.Tensor, use_causal_mask: bool = False) -> torch.Tensor:
        """
        Returns a value V(s) for EACH token in the sequence.
        The input `sequence_tokens` are expected to be just states.
        """
        x = self.embed(sequence_tokens)
        outputs = self.transformer(inputs_embeds=x)
        all_token_outputs = outputs.last_hidden_state
        values = self.head(all_token_outputs)
        return values.squeeze(-1)