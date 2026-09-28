import numpy as np
import pickle
from gwpy.timeseries import TimeSeries
NOISE_FILE = 'h1_noise_processed.hdf5'
TEMPLATES_FILE = 'waveform_templates.pkl'
SEGMENT_LENGTH = 4096       # samples per training example (4 sec at 1024 Hz)
SNR_LEVELS = [5, 8, 12, 16, 20]
print("Loading preprocessed noise...")
noise = TimeSeries.read(NOISE_FILE)
noise_array = np.array(noise)
print(f"  Noise length: {len(noise_array)} samples")

print("Loading waveform templates...")
with open(TEMPLATES_FILE, 'rb') as f:
    templates = pickle.load(f)
print(f"  Loaded {len(templates)} templates\n")
def inject_signal(noise_segment, waveform, target_snr):
    """Combine a noise segment with a scaled waveform to hit a target SNR."""
    wf = np.array(waveform)

    # Match waveform length to segment length (pad or crop)
    if len(wf) < len(noise_segment):
        pad_total = len(noise_segment) - len(wf)
        wf = np.pad(wf, (pad_total // 2, pad_total - pad_total // 2))
    else:
        # Crop to the segment length, centered on the merger (end of array)
        wf = wf[-len(noise_segment):]

    # Approximate amplitude scaling to reach target SNR
    current_snr = np.std(wf) / np.std(noise_segment)
    scale = target_snr / (current_snr + 1e-12)
    wf_scaled = wf * scale

    noisy = noise_segment + wf_scaled
    clean = wf_scaled
    return noisy, clean

X_noisy, Y_clean = [], []

print("Building dataset...")
for idx, template in enumerate(templates):
    waveform = template['waveform']

    for snr in SNR_LEVELS:
        # Pick a random starting point in the noise for variety
        max_start = len(noise_array) - SEGMENT_LENGTH
        start = np.random.randint(0, max_start)
        segment = noise_array[start:start + SEGMENT_LENGTH]

        noisy, clean = inject_signal(segment, waveform, snr)
        X_noisy.append(noisy)
        Y_clean.append(clean)

    if (idx + 1) % 100 == 0:
        print(f"  {idx + 1}/{len(templates)} templates processed")

X_noisy = np.array(X_noisy, dtype=np.float32)
Y_clean = np.array(Y_clean, dtype=np.float32)

print(f"\nDataset built:")
print(f"  X_noisy shape: {X_noisy.shape}")
print(f"  Y_clean shape: {Y_clean.shape}")
np.save('X_noisy.npy', X_noisy)
np.save('Y_clean.npy', Y_clean)
print("\nSaved X_noisy.npy and Y_clean.npy")