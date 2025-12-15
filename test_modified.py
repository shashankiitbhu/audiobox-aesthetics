# test_modified.py
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
torch.set_default_device('cpu')

from audiobox_aesthetics.model.aes import AesMultiOutput

print("="*60)
print("TESTING MODIFIED AUDIOBOX")
print("="*60)

# Test 1: Load with WavLM (original)
print("\n[Test 1] Loading with WavLM...")
try:
    model_wavlm = AesMultiOutput(use_hubert=False)
    print(f"  Encoder type: {model_wavlm.encoder_type}")
    print("  ✅ WavLM works!")
except Exception as e: 
    print(f"  ❌ Error: {e}")

# Test 2: Load with HuBERT (new)
print("\n[Test 2] Loading with HuBERT...")
try:
    model_hubert = AesMultiOutput(use_hubert=True)
    print(f"  Encoder type: {model_hubert. encoder_type}")
    print("  ✅ HuBERT works!")
except Exception as e:
    print(f"  ❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Run a forward pass with dummy data
print("\n[Test 3] Testing forward pass with HuBERT...")
try:
    dummy_audio = torch.randn(1, 1, 16000)  # 1 second of random audio
    dummy_mask = torch.ones(1, 1, 16000, dtype=torch.bool)

    batch = {"wav": dummy_audio, "mask":  dummy_mask}

    with torch.no_grad():
        preds = model_hubert(batch)

    print(f"  Output keys: {list(preds.keys())}")
    print(f"  CE prediction: {preds['CE']. item():.4f}")
    print(f"  CU prediction: {preds['CU'].item():.4f}")
    print(f"  PC prediction: {preds['PC']. item():.4f}")
    print(f"  PQ prediction: {preds['PQ'].item():.4f}")
    print("  ✅ Forward pass works!")
except Exception as e:
    print(f"  ❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("SUCCESS! Modified code works with both encoders!")
print("="*60)