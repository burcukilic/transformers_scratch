import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        
        # Precompute the positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # Even: sin(pos / 10000^{2i/d_model})
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd:  cos(pos / 10000^{2i/d_model})

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):

        _, seq_len, _ = x.shape

        x = x + self.pe[:, :seq_len, :]

        return x
    
class FeedForwardLayer(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int = 128):
        
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)
    

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 4):
        
        super().__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query: torch.Tensor = None, key: torch.Tensor = None, value: torch.Tensor = None, mask: torch.Tensor = None):

        batch_size, q_len, _ = query.shape
        _, kv_len, _ = key.shape
        
        q = self.q_proj(query).view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = q @ k.transpose(-2, -1) / (self.head_dim ** 0.5)

        if mask is not None:

            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, q_len, kv_len)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (B, 1, T_q, T_kv)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(batch_size, q_len, self.embed_dim)

        return self.out_proj(out)
