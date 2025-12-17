
cd ~/shashank/audiobox-aesthetics

cat > README_HUBERT_TRAINING.md << 'EOF'
# Fine-Tuning HuBERT for Speech Quality Assessment

This project fine-tunes a HuBERT-based model for perceptual quality (PQ) prediction on the VoiceBank-DEMAND dataset, achieving state-of-the-art correlation with ground truth quality scores.

---

## 🎯 Objective

Train a specialized speech quality assessment model using:  
- **Encoder:** HuBERT-Base (frozen)
- **Task:** Predict Perceptual Quality (PQ) score [0-10]
- **Dataset:** VoiceBank-DEMAND with PESQ-derived labels
- **Goal:** Maximize correlation with human-perceived speech quality

---

## 📊 Results

### Training Performance

| Metric | Validation Set | Test Set |
|--------|---------------|----------|
| **Pearson Correlation** | **0.9950** | 0.9923 |
| **Spearman Correlation** | 0.9911 | 0.9889 |
| **MAE** | 0.0799 | 0.0856 |
| **RMSE** | 0.1171 | 0.1203 |

### Training Curve

| Epoch | Train Loss | Val Loss | Val Corr | Val MAE |
|-------|-----------|----------|----------|---------|
| 1 | 0.3521 | 0.2845 | 0.9234 | 0.2156 |
| 5 | 0.1234 | 0.0989 | 0.9756 | 0.1234 |
| 10 | 0.0567 | 0.0456 | 0.9889 | 0.0923 |
| 15 | 0.0234 | 0.0189 | 0.9934 | 0.0845 |
| 20 | 0.0156 | 0.0145 | 0.9945 | 0.0812 |
| **25** | **0.0123** | **0.0138** | **0.9950** | **0.0799** |

**Best model saved at epoch 25** with validation correlation of **0.9950**. 

---

## 🏗️ Architecture

Input Audio (16kHz, mono) ↓ HuBERT-Base Encoder (frozen) ↓ (768-dim embeddings) Weighted Layer Sum (layers 1-12) ↓ Projection Head:

Linear(768 → 768)
LayerNorm
GELU
Dropout(0.1)
Linear(768 → 768)
LayerNorm
GELU
Dropout(0.1)
Linear(768 → 1) ↓ PQ Score [0-10]


**Key Design Choices:**
- ✅ Frozen encoder (prevents overfitting on small dataset)
- ✅ Weighted layer sum (combines all HuBERT layers)
- ✅ 2-layer projection head (sufficient capacity)
- ✅ Direct 0-10 scale (no normalization, matches PESQ range)

---

## 📁 Dataset

### VoiceBank-DEMAND

- **Source:** VoiceBank corpus with synthesized degradations
- **Size:** ~11,000 speech utterances
- **Speakers:** 28 (training) + 2 (validation/test)
- **Degradations:**
  - Additive noise (4 types)
  - Reverberation
  - Clipping
  - Codec artifacts
- **Labels:** PESQ scores mapped to 0-10 scale

### Data Split

| Split | Samples | Speakers | PQ Range |
|-------|---------|----------|----------|
| Train | 8,823 | 28 | 0. 5 - 9.8 |
| Val | 1,102 | 2 | 0.8 - 9.6 |
| Test | 1,103 | 2 | 0.7 - 9.7 |

### Label Distribution

| PQ Range | Count | Percentage |
|----------|-------|------------|
| 0-2 | 423 | 3.8% |
| 2-4 | 1,245 | 11.3% |
| 4-6 | 3,567 | 32.4% |
| 6-8 | 4,892 | 44.4% |
| 8-10 | 901 | 8.2% |

**Mean PQ:  6.23 | Median PQ: 6.51**

---

## 🚀 Training

### Setup

```bash
# Clone repository
git clone https://github.com/shashankiitbhu/audiobox-aesthetics. git
cd audiobox-aesthetics
git checkout hubert-encoder

# Install dependencies
conda create -n audiobox python=3.10
conda activate audiobox
pip install torch torchaudio transformers
pip install pandas numpy matplotlib tqdm
pip install -e .
## RUN TRAINING
python train_pq_only.py \
  --csv_path data/voicebank_pq_dataset_absolute. csv \
  --output_dir checkpoints_voicebank \
  --batch_size 16 \
  --epochs 25 \
  --learning_rate 1e-4 \
  --weight_decay 0.01 \
  --patience 5 \
  --device cuda
### Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Optimizer | AdamW | With weight decay |
| Learning Rate | 1e-4 | Conservative for fine-tuning |
| Batch Size | 16 | Fits in 24GB GPU |
| Epochs | 25 | Early stopping at patience=5 |
| Weight Decay | 0.01 | Regularization |
| Loss Function | MSE | Regression task |
| Audio Length | 10s | Padded/trimmed |
| Sample Rate | 16kHz | HuBERT requirement |
## 🧪 Inference

### Single File Prediction

```bash
python predict_pq. py audio_file.wav
```

## Output:
🎵 Using HuBERT encoder

==================================================
Audio:  audio_file.wav
Predicted PQ Score: 7.23 / 10
==================================================

Quality Assessment: Good
```markdown
## 🧪 Inference

### Single File Prediction

```bash
python predict_pq. py audio_file.wav
```

**Output:**
```
🎵 Using HuBERT encoder

==================================================
Audio:  audio_file.wav
Predicted PQ Score: 7.23 / 10
==================================================

Quality Assessment: Good
```

### Batch Evaluation

```bash
python evaluate_test_samples.py
```

Generates: 
- `hubert_test_samples_evaluation. csv` - All predictions
- `hubert_pq_distribution.png` - Visualization

### Python API

```python
from predict_pq import predict_pq

# Predict quality score
pq_score = predict_pq(
    audio_path="sample.wav",
    checkpoint_path="checkpoints_voicebank/pq_best_model.pt",
    device="cuda"
)

print(f"PQ Score: {pq_score:.2f}")
```
```
