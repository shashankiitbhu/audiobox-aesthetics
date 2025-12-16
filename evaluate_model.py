
#!/usr/bin/env python
"""Evaluate model on test set"""

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from predict_pq import predict_pq

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------
df = pd.read_csv("data/voicebank_pq_dataset_absolute.csv")

# Random 500 samples for quick evaluation
test_df = df.sample(n=min(500, len(df)), random_state=42)

predictions = []
ground_truth = []

print(f"Evaluating on {len(test_df)} samples...")

for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
    try:
        pred_pq = predict_pq(
            row["filepath"],
            "checkpoints_voicebank/pq_best_model.pt",
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        predictions.append(pred_pq)
        ground_truth.append(row["PQ"])
    except Exception as e:
        print(f"Error on {row['filepath']}: {e}")

# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------
predictions = np.array(predictions)
ground_truth = np.array(ground_truth)

correlation = np.corrcoef(predictions, ground_truth)[0, 1]
mae = np.mean(np.abs(predictions - ground_truth))
rmse = np.sqrt(np.mean((predictions - ground_truth) ** 2))

print(f"\n{'='*50}")
print("EVALUATION RESULTS")
print(f"{'='*50}")
print(f"Samples: {len(predictions)}")
print(f"Correlation: {correlation:.4f}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"{'='*50}\n")

# ---------------------------------------------------------------------
# Scatter plot
# ---------------------------------------------------------------------
plt.figure(figsize=(10, 8))
plt.scatter(ground_truth, predictions, alpha=0.5)
plt.plot([0, 10], [0, 10], "r--", label="Perfect prediction")
plt.xlabel("Ground Truth PQ (PESQ)", fontsize=12)
plt.ylabel("Predicted PQ (Model)", fontsize=12)
plt.title(
    f"PQ Prediction Results\nCorrelation: {correlation:.4f}, MAE: {mae:.4f}",
    fontsize=14,
)
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim([0, 10])
plt.ylim([0, 10])
plt.savefig("pq_evaluation.png", dpi=150, bbox_inches="tight")
print("✅ Saved plot: pq_evaluation.png")

# ---------------------------------------------------------------------
# Save detailed results
# ---------------------------------------------------------------------
results_df = pd.DataFrame({
    "filepath": test_df["filepath"].values[:len(predictions)],
    "ground_truth": ground_truth,
    "predicted": predictions,
    "abs_error": np.abs(predictions - ground_truth),
})

results_df.to_csv("pq_evaluation_results.csv", index=False)
print("✅ Saved results: pq_evaluation_results.csv")

