import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from blocks import *

class TransformerDecoderBlock(nn.Module):
    def __init__(self, embed_dim: int = 4, num_heads: int = 2):
        
        super().__init__()
        self.self_attention = MultiHeadAttentionLayer(embed_dim=embed_dim, num_heads=num_heads)
        self.cross_attention = MultiHeadAttentionLayer(embed_dim=embed_dim, num_heads=num_heads) 

        self.feed_forward = FeedForwardLayer(embed_dim=embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.norm4 = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor = None, mask: torch.Tensor = None):
        normed_x = self.norm1(x)
        
        tril_mask = torch.tril(torch.ones((x.size(1), x.size(1)), device=x.device)).bool()

        x = x + self.self_attention(query=normed_x, key=normed_x, value=normed_x, mask=tril_mask)
        
        if encoder_output is not None:
            normed_x = self.norm2(x)
            normed_encoder_output = self.norm3(encoder_output)

            x = x + self.cross_attention(query=normed_x, key=normed_encoder_output, value=normed_encoder_output, mask=mask)

        normed_x = self.norm4(x)
        x = x + self.feed_forward(normed_x)
        
        return x
        

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size: int = 10, embed_dim: int = 2, seq_len: int = 10, num_blocks: int = 2):
        
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim) # (batch_size, seq_len) -> (batch_size, seq_len, embed_dim)
        self.pos_emb = SinusoidalPositionalEncoding(d_model=embed_dim, max_len=seq_len) 
        self.blocks = nn.ModuleList([
            TransformerDecoderBlock(embed_dim=embed_dim)
            for _ in range(num_blocks)
        ])

        self.fc = nn.Linear(embed_dim, vocab_size)
        self.act = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor = None, mask: torch.Tensor = None):
        x = self.token_emb(x)
        x = self.pos_emb(x)

        for block in self.blocks:
            x = block(x, encoder_output=encoder_output, mask=mask)

        x = self.fc(x)
        x = self.act(x)

        return x
    
    def loss(self, x: torch.Tensor, target: torch.Tensor):
        y = self.forward(x)
        y = y.view(-1, y.size(-1))
        target = target.view(-1)
        return F.cross_entropy(y, target)

class DecisionTransformerDecoder(nn.Module):
    def __init__(self, vocab_size: int = 10, embed_dim: int = 2, seq_len: int = 10, num_blocks: int = 2):
        
        super().__init__()
        self.reward_emb = nn.Linear(1, embed_dim)
        self.token_emb = nn.Embedding(vocab_size, embed_dim) # (batch_size, seq_len) -> (batch_size, seq_len, embed_dim)
        self.pos_emb = SinusoidalPositionalEncoding(d_model=embed_dim, max_len=seq_len) 
        self.blocks = nn.ModuleList([
            TransformerDecoderBlock(embed_dim=2*embed_dim)
            for _ in range(num_blocks)
        ])

        self.fc = nn.Linear(2*embed_dim, vocab_size)
        self.act = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, r: torch.Tensor, encoder_output: torch.Tensor = None, mask: torch.Tensor = None):
        x = self.token_emb(x)
        x = self.pos_emb(x)

        r = self.reward_emb(r.unsqueeze(-1))
        x = torch.cat([x, r], dim=-1)  # Concatenate reward embedding
        
        for block in self.blocks:
            x = block(x, encoder_output=encoder_output, mask=mask)

        x = self.fc(x)
        x = self.act(x)

        return x
    
    def loss(self, x: torch.Tensor, r: torch.Tensor, target: torch.Tensor):
        y = self.forward(x, r)
        y = y.view(-1, y.size(-1))
        target = target.view(-1)
        return F.cross_entropy(y, target)
    

class TransformerSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt_input):
        enc_output = self.encoder(src)
        return self.decoder(tgt_input, encoder_output=enc_output)

    def loss(self, src, tgt_input, tgt_output, pad_idx=0):
        logits = self.forward(src, tgt_input)
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt_output.reshape(-1),
            ignore_index=pad_idx
        )