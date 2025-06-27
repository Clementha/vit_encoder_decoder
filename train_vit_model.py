import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Normalize
from torchvision.transforms.functional import to_pil_image
import einops
from multi_digit_dataset import MultiRowGridDigitDataset
from model import LookerTrns, patchify_batch, ViTSeqModel  # ViT encoder
from decoder import CustomTransformerDecoder  # Transformer decoder
import wandb

def train(epochs=20, batch_size=64, lr=5e-5, patch_size=20, num_rows=3, num_cols=3):
    device = torch.device("cpu")
    # if torch.backends.mps.is_available():
    #     device = torch.device("mps")
    # elif torch.cuda.is_available():
    #     device = torch.device("cuda")
    # else:
    #     device = torch.device("cpu")
    # print(f"Using device: {device}")
    
    pad_token = 12
    vocab_size = 13  # 0-9 + <start> + <end> + <pad>

    wandb.init(project="vit-digit-sequence", config={
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "patch_size": patch_size,
        "num_rows": num_rows,
        "num_cols": num_cols,
        "model": "Clement's ViT_Encoder_Decoder_Model",
    })

    dataset = MultiRowGridDigitDataset(num_rows=num_rows, num_cols=num_cols, cell_size=patch_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ViTSeqModel(vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        for i, (img, tgt_seq) in enumerate(dataloader):
            img = img.to(device)                       # [B, 1, H, W]
            tgt_input = tgt_seq[:, :-1].to(device)     # decoder input: [B, T-1]
            tgt_target = tgt_seq[:, 1:].to(device)     # target output: [B, T-1]

            optimizer.zero_grad()
            logits = model(img, tgt_input)             # [B, T-1, vocab_size]
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                tgt_target.reshape(-1),
                ignore_index=pad_token
            )
            loss.backward()
            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

            if i % 10 == 0:
                print(f"Epoch {epoch+1} | Batch {i}/{len(dataloader)} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}, Avg Loss: {avg_loss:.4f}")

        # Just grab one batch for logging
        img, tgt_seq = next(iter(dataloader))
        img = img.to(device)
        tgt_input = tgt_seq[:, :-1].to(device)
        logits = model(img, tgt_input)
        pred = torch.argmax(logits, dim=-1)

        # Only show the first sample in the batch
        example_image = to_pil_image(img[0].cpu())
        pred_label = pred[0].cpu().tolist()
        true_label = tgt_seq[0, 1:].cpu().tolist()  # skip <start>

        wandb.log({"loss": avg_loss, 
                   "grad_norm": model.encoder.emb.weight.grad.norm().item(),
                   "example": wandb.Image(img[0].cpu(), caption=f"Pred: {pred.tolist()} | True: {tgt_seq[0].tolist()}")
                   })

        torch.save(model.state_dict(), f"./checkpoints/vit_seq_epoch{epoch+1}.pth")
        scheduler.step()
    wandb.finish()

if __name__ == "__main__":
    train()