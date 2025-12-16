#!/usr/bin/env python
"""Train HuBERT with PQ head only"""

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torchaudio
from tqdm import tqdm
import numpy as np

from audiobox_aesthetics.model.aes import AesMultiOutput

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------
class AudioPQDataset(Dataset):
    def __init__(self, csv_path, max_duration=10.0, sample_rate=16000):
        self.df = pd.read_csv(csv_path)
        self.max_duration = max_duration
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration * sample_rate)

        logger.info(f"Loaded {len(self.df)} samples")
        logger.info(f"PQ range: {self.df['PQ'].min():.2f} - {self.df['PQ'].max():.2f}")
        logger.info(f"PQ mean: {self.df['PQ'].mean():.2f}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        try:
            waveform, sr = torchaudio.load(row["filepath"])
        except Exception as e:
            logger.warning(f"Failed to load {row['filepath']}: {e}")
            waveform = torch.zeros(1, self.max_samples)
            sr = self.sample_rate

        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if waveform.shape[1] < self.max_samples:
            padding = self.max_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
            mask = torch.ones(1, self.max_samples, dtype=torch.bool)
            mask[0, -padding:] = False
        else:
            waveform = waveform[:, :self.max_samples]
            mask = torch.ones(1, self.max_samples, dtype=torch.bool)

        pq_score = torch.tensor(row["PQ"], dtype=torch.float32)

        return {"wav": waveform, "mask": mask}, pq_score


def collate_fn(batch):
    wavs = torch.stack([item[0]["wav"] for item in batch])
    masks = torch.stack([item[0]["mask"] for item in batch])
    pqs = torch.stack([item[1] for item in batch])
    return {"wav": wavs, "mask": masks}, pqs


# ---------------------------------------------------------------------
# Train / Validate
# ---------------------------------------------------------------------
def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} - Training")
    for batch, pq_labels in pbar:
        batch = {k: v.to(device) for k, v in batch.items()}
        pq_labels = pq_labels.to(device)

        optimizer.zero_grad()

        try:
            preds = model(batch)
            loss = criterion(preds["PQ"].squeeze(), pq_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        except Exception as e:
            logger.error(f"Training error: {e}")

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device, epoch):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch, pq_labels in tqdm(dataloader, desc=f"Epoch {epoch} - Validating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            pq_labels = pq_labels.to(device)

            try:
                preds = model(batch)
                loss = criterion(preds["PQ"].squeeze(), pq_labels)
                total_loss += loss.item()

                all_preds.extend(preds["PQ"].squeeze().cpu().tolist())
                all_labels.extend(pq_labels.cpu().tolist())
            except Exception as e:
                logger.error(f"Validation error: {e}")

    if all_preds:
        corr = np.corrcoef(all_preds, all_labels)[0, 1]
        logger.info(f"Validation correlation: {corr:.4f}")

    return total_loss / len(dataloader)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main(args):
    logger.info("=" * 70)
    logger.info("TRAINING PQ HEAD WITH HUBERT")
    logger.info("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    model = AesMultiOutput(
        use_hubert=True,
        freeze_encoder=args.freeze_encoder,
        use_weighted_layer_sum=True,
        proj_num_layer=2,
        proj_dropout=0.1,
        nth_layer=12,
    ).to(device)

    if args.freeze_layers > 0 and not args.freeze_encoder:
        logger.info(f"Freezing first {args.freeze_layers} HuBERT layers")
        for i, layer in enumerate(model.encoder.encoder.layers):
            if i < args.freeze_layers:
                for p in layer.parameters():
                    p.requires_grad = False

    dataset = AudioPQDataset(args.csv_path)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        logger.info(f"\nEpoch {epoch}/{args.epochs}")

        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        val_loss = validate(
            model, val_loader, criterion, device, epoch
        )

        logger.info(
            f"Summary: Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f}"
        )

        scheduler.step(val_loss)

        ckpt_path = output_dir / f"pq_epoch{epoch}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
            },
            ckpt_path,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = output_dir / "pq_best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                },
                best_path,
            )
            logger.info(f"New best model saved: {best_path}")

    logger.info("Training complete.")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train HuBERT PQ model")
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="checkpoints_pq")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--freeze_layers", type=int, default=6)
    parser.add_argument("--freeze_encoder", action="store_true")

    main(parser.parse_args())
