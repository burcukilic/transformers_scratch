import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        
        # Precompute the positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sine to even indices, cosine to odd
        pe[:, 0::2] = torch.sin(position * div_term)  # Even: sin(pos / 10000^{2i/d_model})
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd:  cos(pos / 10000^{2i/d_model})

        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model), ready to broadcast over batch
        self.register_buffer('pe', pe)  # buffer = non-trainable but moves with model.to(device)

    def forward(self, x: torch.Tensor):

        _, seq_len, _ = x.shape

        x = x + self.pe[:, :seq_len, :]

        return x
    