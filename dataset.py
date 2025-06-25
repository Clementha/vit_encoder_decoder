#
#
#
import torch
import torchvision as tv
import matplotlib.pyplot as plt
import random
import einops


#
#
#
torch.manual_seed(47)
random.seed(47)


#
#
#
class Classify(torch.utils.data.Dataset):
  def __init__(self, train=True):
    super().__init__()
    self.tf = tv.transforms.Compose([tv.transforms.ToTensor(), tv.transforms.Normalize((0.1307,), (0.3081,))])
    self.ds = tv.datasets.MNIST(root='.', train=train, download=True)
    self.ti = tv.transforms.ToPILImage()
    self.ln = len(self.ds)

  def __len__(self):
    return len(self.ds)

  def __getitem__(self, idx):
    img, label = self.ds[idx]
    tnsrs = self.tf(img.resize((56, 56))).squeeze()
    patch = einops.rearrange(tnsrs, '(h ph) (w pw) -> (h w) ph pw', ph=14, pw=14)
    return tnsrs, patch, label


#
#
#
if __name__ == "__main__":

  dc = Classify()
  img, pch, lbl = dc[0]
  print('img', img.shape)
  print('pch', pch.shape)
  print('lbl', lbl) # 5
  plt.imshow(dc.ti(img)); plt.show()
  plt.imshow(dc.ti(einops.rearrange(pch, 'p h w -> h (p w)'))); plt.show()