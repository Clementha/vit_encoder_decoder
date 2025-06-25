#
#
#
import torch


#
#
#
class Attention(torch.nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.dim = dim
    self.qkv_prj = torch.nn.Linear(dim, 3 * dim)
    self.out_prj = torch.nn.Linear(dim, dim)
    self.dropout = torch.nn.Dropout(0.1)

  def forward(self, x):
    qry, key, val = self.qkv_prj(x).split(self.dim, dim=-1)
    scale = self.dim ** -0.5
    att = torch.matmul(qry, key.transpose(-2, -1)) * scale
    att = torch.softmax(att, dim=-1)
    att = self.dropout(att)
    out = torch.matmul(att, val)
    return self.out_prj(out)


#
#
#
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


#
#
#
#
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


#
#
#
class LookerTrns(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.cls = torch.nn.Parameter(torch.randn(1, 1, 128))
    self.emb = torch.nn.Linear(196, 128)
    self.pos = torch.nn.Embedding(17, 128)
    self.register_buffer('rng', torch.arange(17))
    self.enc = torch.nn.ModuleList([EncoderLayer(128) for _ in range(6)])
    self.fin = torch.nn.Sequential(torch.nn.LayerNorm(128), torch.nn.Linear(128, 10))

  def forward(self, x):
    B = x.shape[0]                      # [B, 16, 196]
    pch = self.emb(x)                   # [B, 16, 128]
    cls = self.cls.expand(B, -1, -1)    # [B, 1, 128]
    hdn = torch.cat([cls, pch], dim=1)  # [B, 17, 128]
    hdn = hdn + self.pos(self.rng)      # [B, 17, 128]
    for enc in self.enc: hdn = enc(hdn) # [B, 17, 128]
    out = hdn[:, 0, :]                  # [B, 128]
    return self.fin(out)                # [B, 10]


#
#
#
if __name__ == '__main__':
  x = torch.randn(2, 16, 196)
  mdl = LookerTrns()
  out = mdl(x)
  print(out.shape)