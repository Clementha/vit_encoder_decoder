import torch
import einops
from decoder import MultiHeadAttention, FeedForward
from torch import nn as nn

class FFN(torch.nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.one = torch.nn.Linear(dim, dim)
    self.drp = torch.nn.Dropout(0.1)
    self.rlu = torch.nn.ReLU(inplace=True)
    self.two = torch.nn.Linear(dim, dim)

  def forward(self, x):
    x = self.one(x)
    x = self.rlu(x)
    x = self.drp(x)
    x = self.two(x)
    return x

# Single-head attention is not efficient for large dimensions, so we use multi-head attention
class Attention(torch.nn.Module):
  def __init__(self, dim, num_heads=8):
    super().__init__()
    assert dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
    
    self.dim = dim
    self.num_heads = num_heads
    self.head_dim = dim // num_heads

    self.q_proj = torch.nn.Linear(dim, dim)
    self.k_proj = torch.nn.Linear(dim, dim)
    self.v_proj = torch.nn.Linear(dim, dim)
    self.o_proj = torch.nn.Linear(dim, dim)
    self.dropout = torch.nn.Dropout(0.1)

  def forward(self, x):
    B, N, _ = x.shape  # Batch size, Sequence length, Embedding dim

    # Linear projections
    q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, heads, N, head_dim]
    k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
    v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

    # Scaled Dot-Product Attention
    scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, heads, N, N]
    attn = torch.softmax(scores, dim=-1)
    attn = self.dropout(attn)

    out = attn @ v  # [B, heads, N, head_dim]
    out = out.transpose(1, 2).contiguous().view(B, N, self.dim)  # [B, N, dim]

    return self.o_proj(out)

# This is original ingle-head attention class
"""
class Attention(torch.nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.dim = dim
    self.q_proj = torch.nn.Linear(dim, dim)
    self.k_proj = torch.nn.Linear(dim, dim)
    self.v_proj = torch.nn.Linear(dim, dim)
    self.o_proj = torch.nn.Linear(dim, dim)
    self.drpout = torch.nn.Dropout(0.1)

  def forward(self, x):
    qry = self.q_proj(x)
    key = self.k_proj(x)
    val = self.v_proj(x)
    att = qry @ key.transpose(-2, -1) * self.dim ** -0.5
    att = torch.softmax(att, dim=-1)
    att = self.drpout(att)
    out = torch.matmul(att, val)
    return self.o_proj(out)
"""

class EncoderLayer(torch.nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.att = Attention(dim)
    self.ffn = FFN(dim)
    self.ini = torch.nn.LayerNorm(dim)
    self.fin = torch.nn.LayerNorm(dim)

  def forward(self, src):
    out = self.att(src)
    src = src + out
    src = self.ini(src)
    out = self.ffn(src)
    src = src + out
    src = self.fin(src)
    return src

class LookerTrns(nn.Module):
  def __init__(self, patch_dim=400, emb_dim=128, num_layers=6, num_tokens=17, num_classes=10):
    super().__init__()
    self.cls = nn.Parameter(torch.randn(1, 1, emb_dim))
    self.emb = nn.Linear(patch_dim, emb_dim)
    self.pos = nn.Embedding(num_tokens, emb_dim)
    self.enc = nn.ModuleList([EncoderLayer(emb_dim) for _ in range(num_layers)])
    self.fin = nn.Sequential(
        nn.LayerNorm(emb_dim),
        nn.Linear(emb_dim, num_classes)
    )

  # For encoder-decoder training
  #memory = self.encoder(patches)
  #
  # For standalone classification
  #cls_logits = self.encoder(patches, classify=True)
  #

  def forward(self, x, classify=False):
    B = x.shape[0]                      # [B, N, patch_dim]
    pch = self.emb(x)                   # [B, N, emb_dim]
    cls = self.cls.expand(B, 1, -1)     # [B, 1, emb_dim]
    hdn = torch.cat([cls, pch], dim=1)  # [B, N+1, emb_dim]

    B, N, _ = hdn.shape
    pos_ids = torch.arange(N, device=hdn.device).unsqueeze(0).expand(B, N)
    hdn = hdn + self.pos(pos_ids)

    for enc in self.enc:
        hdn = enc(hdn)

    if classify:
        return self.fin(hdn[:, 0])  # classification via CLS
    else:
        return hdn  # full sequence for decoder

class DecoderBlock(torch.nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.masked_attn = MultiHeadAttention(dim, num_heads)
        self.cross_attn = MultiHeadAttention(dim, num_heads)
        self.ffn = FeedForward(dim)

        self.norm1 = torch.nn.LayerNorm(dim)
        self.norm2 = torch.nn.LayerNorm(dim)
        self.norm3 = torch.nn.LayerNorm(dim)
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, tgt, memory, tgt_mask=None):
        # Masked Self-Attention
        x = self.norm1(tgt + self.masked_attn(tgt, mask=tgt_mask))
        # Encoder-Decoder Attention
        x = self.norm2(x + self.cross_attn(x, context=memory))
        # Feedforward
        x = self.norm3(x + self.ffn(x))
        return x

class CustomTransformerDecoder(torch.nn.Module):
    def __init__(self, vocab_size, dim=128, num_heads=8, num_layers=6, max_len=50):
        super().__init__()
        self.token_emb = torch.nn.Embedding(vocab_size, dim)
        self.pos_emb = torch.nn.Embedding(max_len, dim)
        self.blocks = torch.nn.ModuleList([DecoderBlock(dim, num_heads) for _ in range(num_layers)])
        self.out_proj = torch.nn.Linear(dim, vocab_size)

    def forward(self, tgt_seq, memory):
        B, T = tgt_seq.shape
        positions = torch.arange(T, device=tgt_seq.device).unsqueeze(0).expand(B, T)
        x = self.token_emb(tgt_seq) + self.pos_emb(positions)

        # Create causal mask: [T, T] with upper triangle = -inf
        mask = torch.tril(torch.ones(T, T, device=tgt_seq.device)).unsqueeze(0).unsqueeze(1)
        
        for block in self.blocks:
            x = block(x, memory, tgt_mask=mask)

        return self.out_proj(x)

class ViTSeqModel(torch.nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = LookerTrns(patch_dim=400, emb_dim=128, num_tokens=17)
        self.decoder = CustomTransformerDecoder(vocab_size=vocab_size, dim=128, max_len=12)

    def forward(self, image_tensor, tgt_seq):
        # image_tensor: [B, 1, H, W]
        patches = patchify_batch(image_tensor, patch_size=20)  # [B, N, 400]
        memory = self.encoder(patches)                         # [B, N+1, 128]
        return self.decoder(tgt_seq, memory)                   # [B, T, vocab_size]

def patchify_batch(img_batch, patch_size=14):
    """
    img_batch: [B, 1, H, W]
    Returns: [B, num_patches, patch_dim]
    """
    B, C, H, W = img_batch.shape
    assert C == 1, "Only single-channel images supported"
    assert H % patch_size == 0 and W % patch_size == 0
    img_batch = img_batch.squeeze(1)  # [B, H, W]
    patches = einops.rearrange(img_batch, 'b (h ph) (w pw) -> b (h w) (ph pw)', ph=patch_size, pw=patch_size)
    return patches  # shape: [B, num_patches, patch_dim]

if __name__ == '__main__':
  x = torch.randn(2, 16, 196)
  mdl = LookerTrns()
  out = mdl(x)
  print(out.shape)