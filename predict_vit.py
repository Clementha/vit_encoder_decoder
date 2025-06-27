import torch
from torchvision.utils import save_image
from model import ViTSeqModel
from multi_digit_dataset import MultiRowGridDigitDataset
import glob
import os
import re

if torch.backends.mps.is_available():
        device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
#print(f"Using device: {device}")

# --- Config ---
model_dir = "./checkpoints"  
output_dir = "test_samples"
os.makedirs(output_dir, exist_ok=True)

# --- Load dataset and sample ---
dataset = MultiRowGridDigitDataset(num_rows=3, num_cols=3, cell_size=20)
img, seq = dataset[0]

# --- Save the input image with a constant filename ---
img_path = os.path.join(output_dir, "test_sample.png")
save_image(img, img_path)
print(f"Saved input image to {img_path}")

# --- Print ground truth sequence ---
vocab_map = {10: "<start>", 11: "<end>", 12: "<pad>"}
for d in range(10):
    vocab_map[d] = str(d)

decoded_gt = [vocab_map[t.item()] for t in seq]
print("Ground truth:", decoded_gt)

# --- Sort the model checkpoints to find latest to load ---
def extract_epoch(filename):
    match = re.search(r'epoch(\d+)', filename)
    return int(match.group(1)) if match else -1

# --- Load model ---
vocab_size = 13
model = ViTSeqModel(vocab_size)
ckpt_files = sorted(glob.glob(os.path.join(model_dir,"vit_seq_epoch*.pth")))
ckpt_files = sorted(ckpt_files, key=extract_epoch)

if ckpt_files:
    latest_ckpt = ckpt_files[-1]
    print(f"Using latest model: {latest_ckpt}")
    model.load_state_dict(torch.load(latest_ckpt, map_location=device))
else:
    raise FileNotFoundError("No checkpoint file matching 'vit_seq_epoch*.pth' found.")

#model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

# --- Prepare input for prediction ---
with torch.no_grad():
    img = img.unsqueeze(0)  # [1, 1, H, W]
    
    model = model.to(device) # If not already on device
    img = img.to(device)

    # Start decoding with <start>
    start_token = 10
    end_token = 11
    pad_token = 12
    max_len = 11  # start + up to 5 digits + end

    tgt_seq = torch.tensor([[start_token]], dtype=torch.long, device=device)
    for _ in range(max_len - 1):
        logits = model(img, tgt_seq)  # [1, T, vocab_size]
        next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
        tgt_seq = torch.cat([tgt_seq, torch.tensor([[next_token]], device=device)], dim=1)
        if next_token == end_token:
            break

# --- Decode prediction ---
pred_tokens = tgt_seq.squeeze().tolist()
pred_decoded = [vocab_map[t] for t in pred_tokens]
print("Predicted:", pred_decoded)