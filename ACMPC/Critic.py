import torch
import torch.nn as nn
import math


class CriticTransformer(nn.Module):
    """
    Rete Critico basata su Transformer.
    Stima il Q-value processando una sequenza storica di stati e azioni.
    """

    def __init__(self, nx: int, nu: int, history_len: int, d_model: int = 128, n_head: int = 4, n_layers: int = 3):
        """
        Inizializza il modulo Critico.

        Args:
            nx (int): Dimensione dello stato.
            nu (int): Dimensione del controllo.
            history_len (int): Lunghezza della sequenza storica da processare.
            d_model (int): Dimensione interna del modello Transformer.
            n_head (int): Numero di "teste" nel multi-head attention.
            n_layers (int): Numero di layer nel Transformer Encoder.
        """
        super().__init__()
        self.history_len = history_len
        self.d_model = d_model
        feature_dim = nx + nu  # La dimensione di ogni elemento nella sequenza

        # 1. Layer di Input Embedding
        # Proietta ogni coppia (stato, azione) nella dimensione del modello
        self.input_embedding = nn.Linear(feature_dim, d_model)

        # 2. Positional Encoding
        # Inietta l'informazione sulla posizione di ogni elemento nella sequenza
        self.pos_encoder = PositionalEncoding(d_model, dropout=0.1, max_len=history_len)

        # 3. Transformer Encoder
        # Il cuore del modello, che processa la sequenza
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_model * 4,  # Configurazione standard
            dropout=0.1,
            batch_first=True  # FONDAMENTALE per gestire input (B, Seq, Feat)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=n_layers)

        # 4. Testa di Output (Head)
        # Prende l'output del Transformer e lo mappa a un singolo Q-value
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)  # Output scalare
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """
        Esegue il forward pass del critico.

        Args:
            history (torch.Tensor): Tensore contenente la sequenza di stati e azioni.
                                    Shape: (B, history_len, nx + nu)

        Returns:
            torch.Tensor: Il Q-value stimato, shape (B, 1).
        """
        # Proietta l'input nella dimensione del modello
        embedded = self.input_embedding(history)

        # Aggiunge il positional encoding
        embedded_pos = self.pos_encoder(embedded)

        # Passa attraverso il Transformer
        transformer_out = self.transformer_encoder(embedded_pos)

        # Usiamo solo l'output dell'ultimo token (il più recente) per la stima
        last_token_representation = transformer_out[:, -1, :]

        # Calcola il Q-value finale
        q_value = self.output_head(last_token_representation)

        return q_value


class PositionalEncoding(nn.Module):
    """
    Helper class per il Positional Encoding, un componente standard dei Transformer.
    Preso dalla documentazione ufficiale di PyTorch.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # Adatta il positional encoding alla shape dell'input
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)
