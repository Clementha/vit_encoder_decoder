import torch
import dataset
import model


torch.manual_seed(47)
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vit = model.LookerTrns().to(dev)
vit.load_state_dict(torch.load('./checkpoints/2025_06_25__17_25_06.50.vit.pth'))
vit.eval()

ds = dataset.Classify(train=False)
dl = torch.utils.data.DataLoader(ds, batch_size=1024, shuffle=False)

tot = 0; pos = 0

for _, pch, lbl in dl:
  pch = torch.flatten(pch, start_dim=2).to(dev)
  out = vit(pch)
  _, prd = torch.max(out.data, 1)
  prd = prd.detach().cpu()
  tot += lbl.size(0)
  pos += (prd == lbl).sum().item()

print(f'Tot: {tot} Pos: {pos} Acc: {pos / tot:.2%}')