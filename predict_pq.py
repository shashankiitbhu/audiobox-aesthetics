
#!/usr/bin/env python
"""Predict PQ score for any audio file"""

import argparse
from pathlib import Path

import torch
import torchaudio

from audiobox_aesthetics.model.aes import AesMultiOutput


def predict_pq(audio_path, checkpoint_path, device="cuda"):
    """Predict PQ score for a single audio file"""

    device = torch.device(device)

    # Load model
    model = AesMultiOutput(
        use_hubert=True,
        freeze_encoder=False,
        use_weighted_layer_sum=True,
        proj_num_layer=2,
        proj_dropout=0.1,
        nth_layer=12,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load audio
    waveform, sr = torchaudio.load(audio_path)

    # Resample to 16 kHz
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Pad / trim to 10 seconds
    max_samples = 10 * 16000
    if waveform.shape[1] < max_samples:
        padding = max_samples - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, padding))
        mask = torch.ones(1, max_samples, dtype=torch.bool)
        mask[0, -padding:] = False
    else:
        waveform = waveform[:, :max_samples]
        mask = torch.ones(1, max_samples, dtype=torch.bool)

    # Predict
    with torch.no_grad():
        batch = {
            "wav": waveform.unsqueeze(0).to(device),
            "mask": mask.unsqueeze(0).to(device),
        }
        preds = model(batch)
        pq_score = preds["PQ"].item()

    return pq_score


def main():
    parser = argparse.ArgumentParser(description="Predict PQ score for audio")
    parser.add_argument("audio_path", type=str, help="Path to audio file")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints_voicebank/pq_best_model.pt",
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Inference device",
    )
    args = parser.parse_args()

    pq_score = predict_pq(args.audio_path, args.checkpoint, args.device)

    print(f"\n{'='*50}")
    print(f"Audio: {args.audio_path}")
    print(f"Predicted PQ Score: {pq_score:.2f} / 10")
    print(f"{'='*50}\n")

    if pq_score >= 8:
        quality = "Excellent"
    elif pq_score >= 6:
        quality = "Good"
    elif pq_score >= 4:
        quality = "Fair"
    else:
        quality = "Poor"

    print(f"Quality Assessment: {quality}")


if __name__ == "__main__":
    main()

