#!/usr/bin/env python3
"""Test trained HuBERT model"""
import torch
from audiobox_aesthetics.model.aes import AesMultiOutput
import torchaudio

# Load trained model
print("Loading trained HuBERT model...")
model = AesMultiOutput(
    use_hubert=True,
    freeze_encoder=False,
    use_weighted_layer_sum=True,
    proj_num_layer=2,
    proj_dropout=0.1,
    nth_layer=12,
)

checkpoint = torch.load('checkpoints/checkpoint_epoch10.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
model = model.cuda()

print("✅ Model loaded\n")

# Test on sample audio
test_file = "/home/pixaverse.admin/shashank/audiobox-experiment/audio/1-100032-A-0.wav"
print(f"Testing on: {test_file}\n")

# Load and prepare audio
wav, sr = torchaudio. load(test_file)
if sr != 16000:
    resampler = torchaudio. transforms.Resample(sr, 16000)
    wav = resampler(wav)
if wav.shape[0] > 1:
    wav = wav. mean(0, keepdim=True)

# Pad/trim to 10 seconds
max_samples = 16000 * 10
if wav.shape[1] < max_samples:
    wav = torch.nn.functional.pad(wav, (0, max_samples - wav.shape[1]))
else:
    wav = wav[:, :max_samples]

# Create batch
batch = {
    'wav': wav. unsqueeze(0).cuda(),
    'mask': torch.ones(1, 1, max_samples, dtype=torch.bool).cuda()
}

# Predict
with torch.no_grad():
    preds = model(batch)

print("🎵 HuBERT Predictions:")
print(f"  CE:  {preds['CE'].item():.2f}")
print(f"  CU: {preds['CU']. item():.2f}")
print(f"  PC: {preds['PC'].item():.2f}")
print(f"  PQ: {preds['PQ']. item():.2f}")

print("\n✅ Inference successful!")
