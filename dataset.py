import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class GWDenoiseDataset(Dataset):
    def __init__(self, X_path, Y_path):
        self.X = np.load(X_path).astype(np.float32)
        self.Y = np.load(Y_path).astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).unsqueeze(0)  # add channel dim: (1, length)
        y = torch.from_numpy(self.Y[idx]).unsqueeze(0)
        return x, y

if __name__ == "__main__":
    dataset = GWDenoiseDataset('X_noisy.npy', 'Y_clean.npy')
    print(f"Total examples: {len(dataset)}")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    print(f"Train examples: {len(train_ds)}")
    print(f"Val examples: {len(val_ds)}")

    # quick sanity check on shapes
    x_batch, y_batch = next(iter(train_loader))
    print(f"Batch X shape: {x_batch.shape}")
    print(f"Batch Y shape: {y_batch.shape}")