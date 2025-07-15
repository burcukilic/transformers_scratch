import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from blocks import *

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 4):
        
        super().__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        
        batch_size, seq_len, embed_dim = x.shape
        assert embed_dim == self.embed_dim

        qkv = self.qkv(x)

        # (B, T, 3, num_heads, head_dim) → (3, B, num_heads, T, head_dim)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = q @ k.transpose(-2, -1) / (self.head_dim ** 0.5)  # (B, num_heads, T, T)

        if mask is not None:
            # mask shape: (B, 1, 1, T)
            mask = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = attn @ v  # (B, num_heads, T, head_dim)

        # Merge heads: (B, T, embed_dim)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)

        return self.out_proj(out)

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
        x = x + self.self_attention(self.norm1(x), mask=mask)
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
    
class EncoderClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, seq_len, num_blocks, num_heads):
        super().__init__()
        self.encoder = TransformerEncoder(vocab_size, embed_dim, seq_len, num_blocks)
        self.classifier = nn.Linear(embed_dim, 2)  # 3 classes for sentiment

    def masked_mean(self, x: torch.Tensor, mask: torch.Tensor = None):
        mask = mask.unsqueeze(-1)  # (B, T, 1)
        x = x * mask
        return x.sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    def forward(self, x, mask=None):
        x = self.encoder(x, mask=mask)  # (B, T, C)
        x = self.masked_mean(x, mask=mask)   # (B, C)
        logits = self.classifier(x)
        return logits

def generate_batch(batch_size, seq_len, vocab_size):
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    # 1 if x is sorted, 0 otherwise
    y = (x[:, 1:] >= x[:, :-1]).all(dim=1).long()  # Binary classification: sorted or not
    # attention mask is valid for all tokens
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    return x, attention_mask, y

batch_size = 32
seq_len = 8
vocab_size = 20
embed_dim = 32
num_blocks = 2
num_heads = 4

# create a dataset of 100 batches
#train_data = [generate_batch(batch_size, seq_len, vocab_size) for _ in range(100)]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EncoderClassifier(vocab_size, embed_dim, seq_len, num_blocks, num_heads).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for step in range(10000):
    x_batch, attention_mask, y_batch = generate_batch(batch_size, seq_len, vocab_size)
    x_batch, attention_mask, y_batch = x_batch.to(device), attention_mask.to(device), y_batch.to(device)
    logits = model(x_batch, mask=attention_mask)
    loss = criterion(logits, y_batch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        acc = (logits.argmax(dim=1) == y_batch).float().mean()
        print(f"Step {step}: Loss={loss.item():.4f}, Acc={acc.item():.4f}")

model.eval()

with torch.no_grad():
    # Example inference
    x_test, attention_mask, y_test = generate_batch(batch_size=1000, seq_len=seq_len, vocab_size=vocab_size)
    x_test, attention_mask, y_test = x_test.to(device), attention_mask.to(device), y_test.to(device)
    logits = model(x_test, mask=attention_mask)
    predictions = logits.argmax(dim=1)
    print("Accuracy on test set:", (predictions == y_test).float().mean().item())
    #print("Test inputs:", x_test)
    #print("Predictions:", predictions)
    #print("True labels:", y_test)