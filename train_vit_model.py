import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
# from torchvision.transforms import ToTensor, Normalize
# from torchvision.transforms.functional import to_pil_image
import einops
from multi_digit_dataset import MultiRowGridDigitDataset
from model import LookerTrns, patchify_batch, ViTSeqModel  # ViT encoder
from decoder import CustomTransformerDecoder  # Transformer decoder
import wandb

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

def train(epochs=50, batch_size=64, lr=5e-5, patch_size=20, num_rows=3, num_cols=3):


    transform = transforms.ToTensor()
    dataset = MultiRowGridDigitDataset(num_rows=num_rows, num_cols=num_cols, cell_size=patch_size, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    vocab_size = 13
    pad_token = 12
    
    wandb.init(project="vit-digit-sequence", config={
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "patch_size": patch_size,
        "num_rows": num_rows,
        "num_cols": num_cols,
        "model": "Clement's ViT_Encoder_Decoder_Model",
    })

    model = ViTSeqModel(vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        for i, (img, tgt_seq) in enumerate(dataloader):
            total_tokens = 0
            total_correct = 0
            total_seq = 0
            matched_seq = 0

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

            with torch.no_grad():
                pred_tokens = logits.argmax(dim=-1)
                mask = tgt_target != pad_token
                correct = (pred_tokens == tgt_target) & mask
                token_acc = correct.sum().item() / mask.sum().item()
                seq_match = (pred_tokens == tgt_target).masked_fill(~mask, True).all(dim=1)
                seq_acc = seq_match.sum().item() / seq_match.size(0)

            if i % 10 == 0:
                print(f"Epoch {epoch+1} | Batch {i}/{len(dataloader)} | "
                      f"Loss: {loss.item():.4f} | Token Acc: {token_acc:.3f} | Seq Acc: {seq_acc:.3f}")

                wandb.log({
                    "loss": loss.item(),
                    "token_accuracy": token_acc,
                    "sequence_accuracy": seq_acc,
                    "epoch": epoch + 1,
                    "batch": i
                })

        torch.save(model.state_dict(), f"./checkpoints/vit_seq_epoch{epoch+1}.pth")
        scheduler.step()
    wandb.finish()

if __name__ == "__main__":
    train()