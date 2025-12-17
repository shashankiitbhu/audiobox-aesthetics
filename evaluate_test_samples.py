#!/usr/bin/env python
"""
Evaluate ONLY root-level .flac files in test_samples directory
(no recursion, no subfolders)
Generate CSV and distribution plots
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # cloud-safe backend
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

from predict_pq import predict_pq

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
TEST_SAMPLES_DIR = "/home/pixaverse.admin/shashank/test_samples"
CHECKPOINT_PATH = "checkpoints_voicebank/pq_best_model.pt"

OUTPUT_CSV = "hubert_test_samples_evaluation.csv"
OUTPUT_PLOT = "hubert_pq_distribution.png"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------------
print("=" * 70)
print("  HuBERT Model Evaluation on Root-Level Test Samples")
print("=" * 70)
print(f"Device: {DEVICE}")

# ------------------------------------------------------------------
# Collect ONLY root-level .flac files
# ------------------------------------------------------------------
print(f"\nSearching for root-level .flac files in: {TEST_SAMPLES_DIR}")

all_files = sorted([
    os.path.join(TEST_SAMPLES_DIR, f)
    for f in os.listdir(TEST_SAMPLES_DIR)
    if f.lower().endswith(".flac")
    and os.path.isfile(os.path.join(TEST_SAMPLES_DIR, f))
])

print(f"Found {len(all_files)} .flac files\n")

if not all_files:
    raise RuntimeError("❌ No root-level .flac files found!")

# ------------------------------------------------------------------
# Evaluation loop
# ------------------------------------------------------------------
results = []

print("Evaluating samples...\n")
for audio_path in tqdm(all_files):
    try:
        pq_score = predict_pq(
            audio_path,
            CHECKPOINT_PATH,
            device=DEVICE,
        )

        filename = os.path.basename(audio_path)

        if pq_score >= 8:
            quality = "Excellent"
        elif pq_score >= 6:
            quality = "Good"
        elif pq_score >= 4:
            quality = "Fair"
        else:
            quality = "Poor"

        results.append({
            "filename": filename,
            "PQ": float(pq_score),
            "quality": quality,
        })

    except Exception as e:
        print(f"\n❌ Error processing {audio_path}: {e}")

# ------------------------------------------------------------------
# Save CSV
# ------------------------------------------------------------------
df = pd.DataFrame(results)

if df.empty:
    raise RuntimeError("❌ No samples were successfully evaluated!")

df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Saved CSV: {OUTPUT_CSV}")

# ------------------------------------------------------------------
# Statistics
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("  STATISTICS")
print("=" * 70)

print(f"\nTotal samples: {len(df)}")
print(f"Mean PQ:   {df['PQ'].mean():.2f}")
print(f"Median PQ: {df['PQ'].median():.2f}")
print(f"Std PQ:    {df['PQ'].std():.2f}")
print(f"Min PQ:    {df['PQ'].min():.2f}")
print(f"Max PQ:    {df['PQ'].max():.2f}")

print("\nQuality distribution:")
for q, c in df["quality"].value_counts().items():
    print(f"  {q:<10}: {c:>4} ({100*c/len(df):5.1f}%)")

# ------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------
print("\n📊 Generating plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("HuBERT PQ Evaluation (Root-Level test_samples)", fontsize=16)

# Histogram
axes[0, 0].hist(df["PQ"], bins=20, alpha=0.7)
axes[0, 0].axvline(df["PQ"].mean(), linestyle="--", label="Mean")
axes[0, 0].axvline(df["PQ"].median(), linestyle="--", label="Median")
axes[0, 0].set_title("PQ Distribution")
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Boxplot
axes[0, 1].boxplot(df["PQ"], vert=True)
axes[0, 1].set_title("PQ Box Plot")
axes[0, 1].grid(alpha=0.3)

# Quality bar chart
quality_counts = df["quality"].value_counts()
axes[1, 0].bar(quality_counts.index, quality_counts.values)
axes[1, 0].set_title("Quality Categories")
axes[1, 0].grid(alpha=0.3)

# KDE
df["PQ"].plot.kde(ax=axes[1, 1], linewidth=2)
axes[1, 1].set_title("PQ Density")
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=300)
print(f"✅ Saved plot: {OUTPUT_PLOT}")

# ------------------------------------------------------------------
# Top / Bottom samples
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("TOP 10 HIGHEST PQ")
print(df.nlargest(10, "PQ")[["filename", "PQ"]].to_string(index=False))

print("\n" + "=" * 70)
print("TOP 10 LOWEST PQ")
print(df.nsmallest(10, "PQ")[["filename", "PQ"]].to_string(index=False))

print("\n" + "=" * 70)
print("✅ Evaluation Complete")
print("=" * 70)
