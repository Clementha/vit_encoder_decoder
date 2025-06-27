import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision import transforms
import random
from PIL import Image, ImageOps
import numpy as np

class MultiRowGridDigitDataset(Dataset):
    def __init__(self, num_rows=1, num_cols=5, cell_size=20, max_digits=None, transform=None):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.cell_size = cell_size
        self.max_digits = max_digits if max_digits is not None else num_rows * num_cols
        self.transform = transform if transform else transforms.ToTensor()

        self.mnist = MNIST(root="./data", train=True, download=True)
        self.pad_token = 12
        self.start_token = 10
        self.end_token = 11

    def __len__(self):
        return 10000

    def __getitem__(self, idx):
        grid_img = Image.new("L", (self.num_cols * self.cell_size, self.num_rows * self.cell_size), color=0)

        num_digits = random.randint(1, self.max_digits)
        digit_indices = random.choices(range(len(self.mnist)), k=num_digits)
        labels = [self.mnist[i][1] for i in digit_indices]

        total_cells = self.num_rows * self.num_cols
        cell_indices = list(range(total_cells))
        random.shuffle(cell_indices)
        selected_cells = cell_indices[:num_digits]

        cell_and_label = []
        for digit_idx, cell_id in zip(digit_indices, selected_cells):
            digit_img = self.mnist[digit_idx][0]
            digit_resized = ImageOps.fit(digit_img, (self.cell_size, self.cell_size))

            row = cell_id // self.num_cols
            col = cell_id % self.num_cols
            x = col * self.cell_size
            y = row * self.cell_size
            grid_img.paste(digit_resized, (x, y))

            label = self.mnist[digit_idx][1]
            cell_and_label.append((cell_id, label))

        # Sort by cell position (top-to-bottom, left-to-right)
        cell_and_label.sort()
        labels = [label for _, label in cell_and_label]


        img_tensor = self.transform(grid_img)  # [1, H, W]

        tokens = [self.start_token] + labels + [self.end_token]
        pad_len = self.max_digits + 2 - len(tokens)
        tokens += [self.pad_token] * pad_len
        tgt_seq = torch.tensor(tokens, dtype=torch.long)

        return img_tensor, tgt_seq


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