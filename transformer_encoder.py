import torch
import torch.nn as nn
from blocks import *

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim: int = 2):
        
        super().__init__()

        self.self_attention = MultiHeadAttentionLayer(embed_dim=embed_dim, num_heads=2)
        self.feed_forward = FeedForwardLayer(embed_dim=embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        # In the original 'Attention is All you need' paper, firstly attention/ff are then, and then the linear norm is applied,
        # However since it is more stable to perform linear norm first, modern GPT/Bert does this ->
        normed_x = self.norm1(x)
        x = x + self.self_attention(query=normed_x, key=normed_x, value=normed_x, mask=mask)
        x = x + self.feed_forward(self.norm2(x))

        return x
    
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size: int = 10, embed_dim: int = 2, seq_len: int = 10, num_blocks: int = 2):
        
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, embed_dim) # (batch_size, seq_len) -> (batch_size, seq_len, embed_dim)
        self.pos_emb = SinusoidalPositionalEncoding(d_model=embed_dim, max_len=seq_len) 
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim=embed_dim)
            for _ in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        x = self.token_emb(x)
        x = self.pos_emb(x)

        for block in self.blocks:
            x = block(x, mask=mask)

        return x

