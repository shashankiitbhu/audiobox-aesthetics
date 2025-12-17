
#!/usr/bin/env python
"""Test on files that were definitely NOT used in training"""

import pandas as pd
import torch
from predict_pq import predict_pq

# ---------------------------------------------------------------------
# Load training dataset
# ---------------------------------------------------------------------
df_train = pd.read_csv("data/voicebank_pq_dataset_absolute.csv")
train_files = set(df_train["filepath"].values)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------
# Clean samples (should have high PQ)
# ---------------------------------------------------------------------
clean_samples = [
    "/home/pixaverse.admin/shashank/datset_download/clean_trainset/clean_trainset_28spk_wav/p226_001.wav",
    "/home/pixaverse.admin/shashank/datset_download/clean_trainset/clean_trainset_28spk_wav/p226_002.wav",
    "/home/pixaverse.admin/shashank/datset_download/clean_trainset/clean_trainset_28spk_wav/p226_010.wav",
]

print("Testing on CLEAN samples (expected PQ ≈ 9–10):\n")

for sample in clean_samples:
    if sample not in train_files:
        pq = predict_pq(
            sample,
            "checkpoints_voicebank/pq_best_model.pt",
            device=device,
        )
        print(f"✓ {sample.split('/')[-1]}: PQ = {pq:.2f}")
    else:
        print(f"⚠️  {sample} WAS in training set — SKIPPED")

# ---------------------------------------------------------------------
# Noisy samples (should have lower PQ)
# ---------------------------------------------------------------------
noisy_samples = [
    "/home/pixaverse.admin/shashank/datset_download/noisy_trainset/noisy_trainset_28spk_wav/p226_001.wav",
    "/home/pixaverse.admin/shashank/datset_download/noisy_trainset/noisy_trainset_28spk_wav/p226_002.wav",
]

print("\nTesting on NOISY samples (expected PQ ≈ 3–7):\n")

for sample in noisy_samples:
    if sample not in train_files:
        pq = predict_pq(
            sample,
            "checkpoints_voicebank/pq_best_model.pt",
            device=device,
        )
        print(f"✓ {sample.split('/')[-1]}: PQ = {pq:.2f}")
    else:
        print(f"⚠️  {sample} WAS in training set — SKIPPED")

