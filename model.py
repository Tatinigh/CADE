import torch
import torch.nn as nn

class CNNDenoiser(nn.Module):
    def __init__(self):
        super().__init__()

        # ---------------- Encoder ----------------
        self.enc1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.enc3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        # ---------------- Bottleneck ----------------
        self.bottleneck = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        # ---------------- Decoder (with skip connections) ----------------
        self.dec3 = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=9, stride=2, padding=4, output_padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose1d(64, 16, kernel_size=9, stride=2, padding=4, output_padding=1),  # 64 = 32+32 skip
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose1d(32, 1, kernel_size=9, stride=2, padding=4, output_padding=1),   # 32 = 16+16 skip
        )

    def forward(self, x):
        e1 = self.enc1(x)   # downsample 1
        e2 = self.enc2(e1)  # downsample 2
        e3 = self.enc3(e2)  # downsample 3

        b = self.bottleneck(e3)

        d3 = self.dec3(b)
        d3 = torch.cat([d3, e2], dim=1)  # skip connection

        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e1], dim=1)  # skip connection

        out = self.dec1(d2)
        return out

if __name__ == "__main__":
    model = CNNDenoiser()
    x = torch.randn(4, 1, 4096)   # batch of 4, 1 channel, 4096 samples
    out = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")