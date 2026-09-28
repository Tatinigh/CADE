import torch
import numpy as np
from torch.utils.data import DataLoader, random_split

from dataset import GWDenoiseDataset
from model import CNNDenoiser

# ============================================
# CONFIG
# ============================================
MODEL_PATH = 'cnn_denoiser.pt'
BATCH_SIZE = 32

# ============================================
# Load model
# ============================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNNDenoiser().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(f"Loaded model from {MODEL_PATH}")

# ============================================
# Load validation set (same split logic as training)
# ============================================
dataset = GWDenoiseDataset('X_noisy.npy', 'Y_clean.npy')
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

# fix the seed so this matches the same split used in training
torch.manual_seed(42)
train_ds, val_ds = random_split(dataset, [train_size, val_size])

val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# ============================================
# Overlap / Match metric
# ============================================
def compute_overlap(signal_a, signal_b):
    """Normalized cross-correlation overlap (0 to 1). 1.0 = perfect match."""
    a = signal_a / (np.linalg.norm(signal_a) + 1e-12)
    b = signal_b / (np.linalg.norm(signal_b) + 1e-12)
    return np.abs(np.dot(a, b))

# ============================================
# Run evaluation
# ============================================
overlaps = []
mse_losses = []

with torch.no_grad():
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)

        mse = torch.mean((pred - y) ** 2, dim=(1, 2))
        mse_losses.extend(mse.cpu().numpy())

        pred_np = pred.cpu().numpy()
        y_np = y.cpu().numpy()
        for p, t in zip(pred_np, y_np):
            overlaps.append(compute_overlap(p.flatten(), t.flatten()))

overlaps = np.array(overlaps)
mse_losses = np.array(mse_losses)

# ============================================
# Report results
# ============================================
print(f"\n=== Evaluation Results ({len(overlaps)} validation examples) ===")
print(f"Mean overlap:   {overlaps.mean():.4f}")
print(f"Median overlap: {np.median(overlaps):.4f}")
print(f"Min overlap:    {overlaps.min():.4f}")
print(f"Max overlap:    {overlaps.max():.4f}")
print(f"\nMean MSE loss:  {mse_losses.mean():.6f}")

# how many examples achieve a "good" reconstruction (overlap > 0.9)
good_fraction = (overlaps > 0.9).mean()
print(f"\nFraction with overlap > 0.9: {good_fraction:.2%}")