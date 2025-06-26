import torch
import random
import dataset
import model
import matplotlib.pyplot as plt

# Set seed for reproducibility
torch.manual_seed(99)
random.seed(99)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
vit = model.LookerTrns().to(device)
vit.load_state_dict(torch.load("./checkpoints/2025_06_25__16_55_11.5.vit.pth", map_location=device))
vit.eval()

# Load MNIST test dataset
ds = dataset.Classify(train=False)
idx = random.randint(0, len(ds) - 1)
img, patch, label = ds[idx]

# Show the image
plt.imshow(ds.ti(img), cmap="gray")
plt.title(f"True Label: {label}")
plt.axis("off")
plt.show()

# Prepare input for model
patch = patch.unsqueeze(0)  # [1, 16, 14, 14]
patch = patch.flatten(start_dim=2).to(device)  # [1, 16, 196]

# Run prediction
with torch.no_grad():
    logits = vit(patch)         # [1, 10]
    probs = torch.softmax(logits, dim=1).squeeze()  # [10]

# Print scores
print("Digit prediction scores (softmax probabilities):")
for i, score in enumerate(probs):
    print(f"Digit {i}: {score:.4f}")