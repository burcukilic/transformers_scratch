import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import *
from transformer_decoder import *

class DecisionTransformer_DiscreteActions(nn.Module):
    def __init__(self, vocab_size: int = 10, embed_dim: int = 2, seq_len: int = 10, num_blocks: int = 2):
        # Decision Transformer with discrete actions
        super().__init__()
        self.reward_emb = nn.Linear(1, embed_dim)
        self.state_emb = nn.Linear(2, embed_dim)
        self.token_emb = nn.Embedding(vocab_size, embed_dim) # (batch_size, seq_len) -> (batch_size, seq_len, embed_dim)
        self.pos_emb = SinusoidalPositionalEncoding(d_model=2*embed_dim, max_len=seq_len) 
        self.ln = nn.LayerNorm(2 * embed_dim)
        self.blocks = nn.ModuleList([
            TransformerDecoderBlock(embed_dim=2*embed_dim)
            for _ in range(num_blocks)
        ])

        self.fc = nn.Linear(2*embed_dim, vocab_size)
        self.act = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, r: torch.Tensor, s: torch.Tensor, encoder_output: torch.Tensor = None, mask: torch.Tensor = None):
        x_token = self.token_emb(x)          # (B, T, D)
        r_emb = self.reward_emb(r.unsqueeze(-1))  # (B, T, D)
        #s_emb = self.state_emb(s)            # (B, T, D)
        x = torch.cat([x_token, r_emb], dim=-1)  # (B, T, 3D)
        x = self.ln(x)  # LayerNorm over the concatenated embeddings
        x = self.pos_emb(x)  # positional encoding over full input
                
        for block in self.blocks:
            x = block(x, encoder_output=encoder_output, mask=mask)

        x = self.fc(x)
        #x = self.act(x)

        return x
    
    def loss(self, x: torch.Tensor, r: torch.Tensor, s: torch.Tensor, target: torch.Tensor):
        y = self.forward(x, r, s)
        y = y.view(-1, y.size(-1))
        target = target.view(-1)
        return F.cross_entropy(y, target, ignore_index=-1)
    
class DecisionTransformer(nn.Module):
    def __init__(self, action_dim: int = None, state_dim: int = None, embed_dim: int = 2, seq_len: int = 10, num_blocks: int = 2):
        # Normal Decision Transformer with Continuous actions
        super().__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.embed_dim = embed_dim

        self.reward_emb = nn.Linear(1, embed_dim)
        self.state_emb = nn.Linear(state_dim, embed_dim)
        self.action_emb = nn.Linear(action_dim, embed_dim)

        self.pos_emb = SinusoidalPositionalEncoding(d_model=3 * embed_dim, max_len=seq_len)
        self.ln = nn.LayerNorm(3 * embed_dim)

        self.blocks = nn.ModuleList([
            TransformerDecoderBlock(embed_dim=3 * embed_dim)
            for _ in range(num_blocks)
        ])

        self.fc = nn.Linear(3 * embed_dim, action_dim)


    def forward(self, a: torch.Tensor, r: torch.Tensor, s: torch.Tensor, encoder_output: torch.Tensor = None):
        a_emb = self.action_emb(a)               # (B, T, D)
        r_emb = self.reward_emb(r)               # (B, T, D)
        s_emb = self.state_emb(s)                # (B, T, D)

        x = torch.cat([a_emb, r_emb, s_emb], dim=-1)  # (B, T, 3D)
        x = self.ln(x)
        x = self.pos_emb(x)

        for block in self.blocks:
            x = block(x)

        out = self.fc(x)  # (B, T, action_dim)
        return out
    
    def loss(self, a: torch.Tensor, r: torch.Tensor, s: torch.Tensor, target: torch.Tensor):
        pred = self.forward(a, r, s)
        return F.mse_loss(pred, target)

class DiscreteDecisionTransformer(nn.Module):
    # Decision Transformer with Continuous actions Conditioned with task
    def __init__(self, action_dim: int = None, state_dim: int = None, task_dim: int = 2, embed_dim: int = 2, seq_len: int = 10, num_blocks: int = 2):
        
        super().__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.task_dim = task_dim
        self.embed_dim = embed_dim

        self.reward_emb = nn.Linear(1, embed_dim)
        self.state_emb = nn.Linear(state_dim, embed_dim)
        self.action_emb = nn.Linear(action_dim, embed_dim)
        self.task_emb = nn.Embedding(task_dim, 3*embed_dim)

        self.pos_emb = SinusoidalPositionalEncoding(d_model=3 * embed_dim, max_len=seq_len)
        self.ln = nn.LayerNorm(3 * embed_dim)

        self.blocks = nn.ModuleList([
            TransformerDecoderBlock(embed_dim=3 * embed_dim)
            for _ in range(num_blocks)
        ])

        self.fc = nn.Linear(3 * embed_dim, action_dim)


    def forward(self, a: torch.Tensor, r: torch.Tensor, s: torch.Tensor, t: torch.Tensor):
        a_emb = self.action_emb(a)
        r_emb = self.reward_emb(r)
        s_emb = self.state_emb(s)
        t_emb = self.task_emb(t)

        x = torch.cat([a_emb, r_emb, s_emb], dim=-1)  # (B, T, 4D)

        x = self.ln(x)
        x = self.pos_emb(x)

        for block in self.blocks:
            x = block(x, encoder_output=t_emb)

        out = self.fc(x)  # (B, T, action_dim)
        return out
    
    def loss(self, a: torch.Tensor, r: torch.Tensor, s: torch.Tensor, t: torch.Tensor, target: torch.Tensor):
        pred = self.forward(a, r, s, t)
        return F.mse_loss(pred, target)