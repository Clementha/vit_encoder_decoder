import tqdm
import torch
import datetime
import dataset
import model
import wandb

torch.manual_seed(42)
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ts = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
torch.os.makedirs('./checkpoints', exist_ok=True)

vit = model.LookerTrns().to(dev)
torch.save(vit.state_dict(), f'./checkpoints/{ts}.0.vit.pth')
print('vit:', sum(p.numel() for p in vit.parameters())) # 626,570
opt = torch.optim.Adam(vit.parameters(), lr=0.001)
crt = torch.nn.CrossEntropyLoss()
wandb.init(project='mlx8-week-03-vit')

ds = dataset.Classify()
dl = torch.utils.data.DataLoader(ds, batch_size=1024, shuffle=True)

for epoch in range(50):
  prgs = tqdm.tqdm(dl, desc=f"Epoch {epoch + 1}", leave=False)
  for idx, (_, ptch, lbls) in enumerate(prgs):
    ptch = torch.flatten(ptch, start_dim=2).to(dev)
    lbls = lbls.to(dev)
    outs = vit(ptch)
    opt.zero_grad()
    loss = crt(outs, lbls)
    loss.backward()
    opt.step()
    wandb.log({'loss': loss.item()})
  torch.save(vit.state_dict(), f'./checkpoints/{ts}.{epoch + 1}.vit.pth')

wandb.finish()
