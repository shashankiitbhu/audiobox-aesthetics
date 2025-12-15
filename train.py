
#!/usr/bin/env python3
"""
Train HuBERT-based audiobox-aesthetics model on labeled dataset
"""
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torchaudio
from tqdm import tqdm
from pathlib import Path
import logging

from audiobox_aesthetics. model. aes import AesMultiOutput, Normalize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dataset
class AudioAestheticsDataset(Dataset):
    def __init__(self, csv_path, max_duration=10.0, sample_rate=16000):
        self.df = pd.read_csv(csv_path)
        self.max_duration = max_duration
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration * sample_rate)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load audio
        waveform, sr = torchaudio.load(row['filepath'])
        
        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio. transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Pad or trim to max_duration
        if waveform.shape[1] < self.max_samples:
            # Pad
            padding = self.max_samples - waveform.shape[1]
            waveform = torch.nn. functional.pad(waveform, (0, padding))
            mask = torch.ones(1, self.max_samples, dtype=torch.bool)
            mask[0, -padding:] = False
        else:
            # Trim
            waveform = waveform[: , :self.max_samples]
            mask = torch.ones(1, self.max_samples, dtype=torch.bool)
        
        # Labels
        labels = {
            'CE': torch.tensor(row['CE'], dtype=torch.float32),
            'CU': torch.tensor(row['CU'], dtype=torch.float32),
            'PC': torch.tensor(row['PC'], dtype=torch.float32),
            'PQ': torch.tensor(row['PQ'], dtype=torch.float32),
        }
        
        return {
            'wav': waveform,
            'mask': mask,
            'labels':  labels
        }

def collate_fn(batch):
    """Collate function for DataLoader"""
    wavs = torch.stack([item['wav'] for item in batch])
    masks = torch.stack([item['mask'] for item in batch])
    
    labels = {
        'CE': torch.stack([item['labels']['CE'] for item in batch]),
        'CU': torch.stack([item['labels']['CU'] for item in batch]),
        'PC': torch.stack([item['labels']['PC'] for item in batch]),
        'PQ': torch.stack([item['labels']['PQ'] for item in batch]),
    }
    
    return {'wav': wavs, 'mask': masks}, labels

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch, labels in pbar:
        # Move to device
        batch = {k: v.  to(device) for k, v in batch.items()}
        labels = {k: v. to(device) for k, v in labels.items()}
        
        # Forward
        optimizer.zero_grad()
        preds = model(batch)
        
        # Compute loss for each axis
        loss = 0
        for axis in ['CE', 'CU', 'PC', 'PQ']:
            loss += criterion(preds[axis], labels[axis])
        
        # Backward
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader)

def main(args):
    logger.info("=" * 70)
    logger.info("TRAINING AUDIOBOX-AESTHETICS WITH HUBERT")
    logger.info("=" * 70)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    # Create model
    logger.info("\n[1] Initializing model...")
    model = AesMultiOutput(
        use_hubert=True,
        freeze_encoder=False,  # Train encoder
        use_weighted_layer_sum=True,
        proj_num_layer=2,
        proj_dropout=0.1,
        nth_layer=12,
    )
    model = model.to(device)
    logger.info("    ✅ Model initialized with HuBERT encoder")
    
    # Optionally freeze first N layers
    if args.freeze_layers > 0:
        logger.info(f"    Freezing first {args.freeze_layers} HuBERT layers")
        for i, layer in enumerate(model.encoder.encoder.layers):
            if i < args.freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False
    
    # Dataset
    logger.info(f"\n[2] Loading dataset from {args.csv_path}...")
    dataset = AudioAestheticsDataset(args.csv_path)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    logger.info(f"    ✅ Loaded {len(dataset)} samples")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=args.lr,
        weight_decay=args. weight_decay
    )
    criterion = nn.MSELoss()
    
    # Training loop
    logger. info(f"\n[3] Training for {args.epochs} epochs..  .\n")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    for epoch in range(args.epochs):
        avg_loss = train_epoch(model, dataloader, optimizer, criterion, device)
        logger.info(f"Epoch {epoch+1}/{args.epochs}:  Loss = {avg_loss:.4f}")
        
        # Save checkpoint
        checkpoint_path = output_dir / f"checkpoint_epoch{epoch+1}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer. state_dict(),
            'loss': avg_loss,
            'config': {
                'use_hubert': True,
                'freeze_encoder': False,
                'use_weighted_layer_sum': True,
                'proj_num_layer': 2,
                'proj_dropout': 0.1,
                'nth_layer': 12,
            }
        }, checkpoint_path)
        logger.info(f"    💾 Saved:  {checkpoint_path}")
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ TRAINING COMPLETE!")
    logger.info("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True, help="Path to labels CSV")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--freeze_layers", type=int, default=6, help="Number of HuBERT layers to freeze")
    
    args = parser.parse_args()
    main(args)
