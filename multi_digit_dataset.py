import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision import transforms
import random
from PIL import Image, ImageOps
import numpy as np

class MultiRowGridDigitDataset(Dataset):
    def __init__(self, num_rows=3, num_cols=3, cell_size=20, length=10000, transform=None):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.cell_size = cell_size
        self.max_digits = num_rows * num_cols
        self.length = length
        self.transform = transform if transform else transforms.ToTensor()

        self.mnist = MNIST(root="./data", train=True, download=True)
        self.pad_token = 12
        self.start_token = 10
        self.end_token = 11

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Decide number of real digits (1 to max_digits)
        num_digits = random.randint(1, self.max_digits)

        # Pick N digit samples (with replacement)
        digit_indices = random.choices(range(len(self.mnist)), k=num_digits)
        digits = [self.mnist[i] for i in digit_indices]
        digit_imgs = [img.resize((self.cell_size, self.cell_size), Image.BILINEAR) for img, _ in digits]
        digit_labels = [label for _, label in digits]

        # Prepare full grid: fill with blanks
        blank_img = Image.new("L", (self.cell_size, self.cell_size), color=0)
        full_imgs = [blank_img for _ in range(self.max_digits)]

        # Randomly assign digit positions
        positions = list(range(self.max_digits))
        digit_positions = random.sample(positions, k=num_digits)

        for img, pos in zip(digit_imgs, digit_positions):
            full_imgs[pos] = img

        # Create the grid image
        grid_width = self.num_cols * self.cell_size
        grid_height = self.num_rows * self.cell_size
        grid_img = Image.new("L", (grid_width, grid_height), color=0)

        for i, digit_img in enumerate(full_imgs):
            row = i // self.num_cols
            col = i % self.num_cols
            x = col * self.cell_size
            y = row * self.cell_size
            grid_img.paste(digit_img, (x, y))

        image_tensor = self.transform(grid_img)

        # Track (position, label) pairs
        position_label_pairs = list(zip(digit_positions, digit_labels))

        # Sort by grid layout order: top-left to bottom-right
        position_label_pairs.sort()

        # Extract labels in visual order
        ordered_labels = [label for _, label in position_label_pairs]

        # Final sequence
        label_seq = [self.start_token] + ordered_labels + [self.end_token]
        pad_len = self.max_digits + 2 - len(label_seq)
        label_seq += [self.pad_token] * pad_len
        label_tensor = torch.tensor(label_seq, dtype=torch.long)

        return image_tensor, label_tensor


import matplotlib.pyplot as plt
import os
from torchvision.utils import save_image

if __name__ == '__main__':
    # Directory to save output
    os.makedirs("output_samples", exist_ok=True)

    # Load dataset
    dataset = MultiRowGridDigitDataset(num_rows=3, num_cols=3, cell_size=20)

    # Vocab mapping
    vocab_map = {10: "start", 11: "end", 12: "pad"}
    for d in range(10):
        vocab_map[d] = str(d)

    # Save N samples
    for i in range(20):
        img, seq = dataset[i]

        # Decode sequence to readable string
        decoded = [vocab_map[t.item()] for t in seq]
        filename = "_".join(decoded)

        # Clean up filename (avoid double underscores)
        filename = filename.replace("__", "_").strip("_")[:100]  # trim long names

        # Save image
        path = f"output_samples/{i:03d}_{filename}.png"
        save_image(img, path)
        print(f"Saved {path}")