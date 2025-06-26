import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Normalize
from PIL import Image
import random
import matplotlib.pyplot as plt

class MultiRowGridDigitDataset(torch.utils.data.Dataset):
    def __init__(self, mnist_root='.', num_rows=2, num_cols=5, cell_size=20, train=True):
        self.mnist = MNIST(root=mnist_root, train=train, download=True)
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.max_digits = num_rows * num_cols
        self.cell_size = cell_size
        self.image_width = self.num_cols * self.cell_size
        self.image_height = self.num_rows * self.cell_size

        self.vocab_size = 13  # 0–9 + <start> + <end> + <pad>
        self.pad_token = 12
        self.start_token = 10
        self.end_token = 11

        self.to_tensor = ToTensor()
        self.normalize = Normalize((0.1307,), (0.3081,))

    def __len__(self):
        return 10000

    def __getitem__(self, idx):
        num_digits = random.randint(2, self.max_digits)
        indices = random.choices(range(len(self.mnist)), k=num_digits)

        canvas = Image.new('L', (self.image_width, self.image_height), 0)
        label_seq = []

        for i, img_idx in enumerate(indices):
            digit_img, label = self.mnist[img_idx]
            digit_img = digit_img.resize((self.cell_size, self.cell_size))
            row = i // self.num_cols
            col = i % self.num_cols
            x = col * self.cell_size
            y = row * self.cell_size
            canvas.paste(digit_img, (x, y))
            label_seq.append(label)

        tensor_img = self.to_tensor(canvas)
        tensor_img = self.normalize(tensor_img)

        tokens = [self.start_token] + label_seq + [self.end_token]
        while len(tokens) < self.max_digits + 2:
            tokens.append(self.pad_token)
        tokens = torch.tensor(tokens)

        return tensor_img, tokens

if __name__ == '__main__':

    dataset = MultiRowGridDigitDataset(num_rows=1, num_cols=4)
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i in range(5):
        img, seq = dataset[i]
        axes[i].imshow(img.squeeze(), cmap='gray')
        axes[i].set_title(f"Seq: {seq.tolist()}")
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()
    # Visualise a few samples
    dataset = GridDigitSequenceDataset()
