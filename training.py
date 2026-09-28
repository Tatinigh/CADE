import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import GWDenoiseDataset
from model import CNNDenoiser

# ============================================
# CONFIG
# ============================================
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
N_EPOCHS = 50
MODEL_SAVE_PATH = 'cnn_denoiser.pt'

# ============================================
# Setup
# ============================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

dataset = GWDenoiseDataset('X_noisy.npy', 'Y_clean.npy')
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

model = CNNDenoiser().to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

# ============================================
# Training loop
# ============================================
best_val_loss = float('inf')

for epoch in range(N_EPOCHS):
    model.train()
    train_loss = 0
    for x, y in tqdm(train_loader, desc=f'Epoch {epoch+1}/{N_EPOCHS}'):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            val_loss += criterion(pred, y).item()
    val_loss /= len(val_loader)

    print(f'Epoch {epoch+1}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}')

    # save the best model based on validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f'  -> New best model saved (val_loss={val_loss:.6f})')

print(f'\nTraining complete. Best val_loss: {best_val_loss:.6f}')
print(f'Model saved to {MODEL_SAVE_PATH}')