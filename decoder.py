import torch
import torch.nn as nn

# Special version used for decoder
class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, context=None, mask=None):
        context = x if context is None else context
        B, N, _ = x.shape
        Q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(context).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(context).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / self.head_dim ** 0.5
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = attn @ V
        out = out.transpose(1, 2).contiguous().view(B, N, self.num_heads * self.head_dim)
        return self.o_proj(out)

class FeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 4, dim)
        )
    def forward(self, x):
        return self.net(x)

class DecoderBlock(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.self_attn = MultiHeadAttention(dim, num_heads)
        self.cross_attn = MultiHeadAttention(dim, num_heads)
        self.ff = FeedForward(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x, memory, mask=None):
        x = self.norm1(x + self.self_attn(x, mask=mask))
        x = self.norm2(x + self.cross_attn(x, context=memory))
        x = self.norm3(x + self.ff(x))
        return x

class CustomTransformerDecoder(nn.Module):
    def __init__(self, vocab_size, dim=128, num_layers=6, num_heads=8, max_len=12):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.pos_embedding = nn.Embedding(max_len, dim)
        self.blocks = nn.ModuleList([DecoderBlock(dim, num_heads) for _ in range(num_layers)])
        self.to_logits = nn.Linear(dim, vocab_size)

    def forward(self, x, memory):
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        x = self.embedding(x) + self.pos_embedding(positions)

        # Create causal mask
        mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(1)

        for block in self.blocks:
            x = block(x, memory, mask=mask)

        return self.to_logits(x)