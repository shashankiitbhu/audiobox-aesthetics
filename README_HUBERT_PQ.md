# HuBERT PQ-Only Training

Fine-tuned HuBERT encoder for speech quality (PQ) prediction on VoiceBank-DEMAND dataset. 

## Results

- **Correlation:** 0.9950 on VoiceBank validation set
- **MAE:** 0.0799
- **RMSE:** 0.1171

## Training

```bash
python train_pq_only.py \
  --csv_path data/voicebank_pq_dataset_absolute.csv \
  --output_dir checkpoints_voicebank \
  --batch_size 16 \
  --epochs 25

