import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from multi_digit_dataset import MultiRowGridDigitDataset
from model import LookerTrns
from decoder import CustomTransformerDecoder
from model import patchify_batch, ViTSeqModel


def evaluate(model, dataloader, device, idx_to_token=None):
    model.eval()
    pad_token = 12
    end_token = 11
    correct = 0
    total = 0

    with torch.no_grad():
        for img, tgt_seq in dataloader:
            img = img.to(device)
            B = img.size(0)
            memory = model.encoder(patchify_batch(img, patch_size=20).to(device))

            # Start with <start> token
            generated = torch.full((B, 1), 10, dtype=torch.long, device=device)  # [B, 1]
            finished = torch.zeros(B, dtype=torch.bool, device=device)

            for _ in range(tgt_seq.size(1) - 1):
                logits = model.decoder(generated, memory)  # [B, T, vocab]
                next_token = logits[:, -1].argmax(-1, keepdim=True)  # [B, 1]
                generated = torch.cat([generated, next_token], dim=1)
                finished |= (next_token.squeeze() == end_token)
                if finished.all():
                    break

            for i in range(B):
                pred = generated[i, 1:].tolist()  # exclude <start>
                truth = tgt_seq[i, 1:].tolist()
                pred = [t for t in pred if t != pad_token and t != end_token]
                truth = [t for t in truth if t != pad_token and t != end_token]
                if pred == truth:
                    correct += 1
                total += 1

                if idx_to_token:
                    print(f"GT: {[idx_to_token[t] for t in truth]} | Pred: {[idx_to_token[t] for t in pred]}")
                else:
                    print(f"GT: {truth} | Pred: {pred}")

    print(f"Accuracy: {correct / total:.2%}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = MultiRowGridDigitDataset(num_rows=1, num_cols=4, cell_size=20, train=False)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    model = ViTSeqModel(vocab_size=13).to(device)
    model.load_state_dict(torch.load("best_seq_model.pth"))  # adjust as needed

    evaluate(model, dataloader, device)


if __name__ == "__main__":
    main()
