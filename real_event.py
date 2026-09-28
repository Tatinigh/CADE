import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries

from model import CNNDenoiser

MODEL_PATH = 'cnn_denoiser.pt'
DETECTOR = 'H1'
EVENT_GPS = 1126259462.4   # GW150914 merger time
WINDOW_BEFORE = 16          # seconds before merger
WINDOW_AFTER = 16           # seconds after merger
SEGMENT_LENGTH = 4096       # must match training window size
SAMPLE_RATE = 1024          # must match your preprocessing rate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNNDenoiser().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(f"Loaded model from {MODEL_PATH}")

print(f"Fetching {DETECTOR} strain data around GW150914...")
start = EVENT_GPS - WINDOW_BEFORE
end = EVENT_GPS + WINDOW_AFTER

real_data = TimeSeries.fetch_open_data(DETECTOR, start, end, sample_rate=4096)

print("Preprocessing (whiten, bandpass, resample)...")
white = real_data.whiten(fftlength=4, overlap=2)
filtered = white.bandpass(20, 500)
final = filtered.resample(SAMPLE_RATE)

real_array = np.array(final)
print(f"  Preprocessed length: {len(real_array)} samples")


start_idx = center_idx - SEGMENT_LENGTH // 2
end_idx = start_idx + SEGMENT_LENGTH

if start_idx < 0 or end_idx > len(real_array):
    raise ValueError("Segment window falls outside available data — adjust WINDOW_BEFORE/AFTER.")

segment = real_array[start_idx:end_idx]
print(f"  Extracted segment: {len(segment)} samples")

segment_tensor = torch.from_numpy(segment.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

with torch.no_grad():
    denoised = model(segment_tensor).cpu().numpy().flatten()


plt.figure(figsize=(12, 5))
plt.plot(segment, alpha=0.5, label='Real strain (noisy)', linewidth=0.7)
plt.plot(denoised, label='Model output (denoised)', linewidth=1.2, color='darkorange')
plt.xlabel('Sample index')
plt.ylabel('Strain (whitened)')
plt.title('CADE Denoiser on GW150914')
plt.legend()
plt.tight_layout()
plt.savefig('gw150914_result.png', dpi=150)
print("\nSaved plot: gw150914_result.png")

print("\nDone. Open gw150914_result.png to visually inspect the result.")