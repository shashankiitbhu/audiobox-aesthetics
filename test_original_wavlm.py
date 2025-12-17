
#!/usr/bin/env python
"""Test Meta's original WavLM model on a sample audio file"""

from audiobox_aesthetics.infer import initialize_predictor

# ---------------------------------------------------------------------
# Initialize predictor (pretrained WavLM from Hugging Face)
# ---------------------------------------------------------------------
print("Loading Meta's pretrained WavLM model...")
predictor = initialize_predictor(ckpt=None)

# ---------------------------------------------------------------------
# Test audio
# ---------------------------------------------------------------------
audio_path = "/home/pixaverse.admin/shashank/test_samples/youtube_sample.wav"

print(f"\nTesting on: {audio_path}\n")

# ---------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------
result = predictor.forward([{"path": audio_path}])

print("=" * 60)
print("Meta's Original WavLM Results")
print("=" * 60)
print(f"Raw Result: {result}")

# ---------------------------------------------------------------------
# Extract PQ score
# ---------------------------------------------------------------------
pq_score = result[0].get("PQ", None)

if pq_score is None:
    raise RuntimeError("PQ score not found in WavLM output")

print(f"\nPQ Score (Original WavLM): {pq_score:.2f} / 10")
print("=" * 60)

# ---------------------------------------------------------------------
# Quality interpretation
# ---------------------------------------------------------------------
if pq_score >= 8:
    quality = "Excellent ⭐⭐⭐⭐⭐"
elif pq_score >= 6:
    quality = "Good ⭐⭐⭐⭐"
elif pq_score >= 4:
    quality = "Fair ⭐⭐⭐"
else:
    quality = "Poor ⭐⭐"

print(f"Quality: {quality}\n")

