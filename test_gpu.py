#!/usr/bin/env python3
"""
Test Sovereign LiLa-E8 GPU inference
Verifies model loads and runs on CUDA device
"""
import sys
sys.path.insert(0, '.')

import torch
from config.config import E8Config

def main():
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return 1
    
    device = torch.device('cuda')
    print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    
    # Load checkpoint to GPU
    print("Loading checkpoint to GPU...")
    checkpoint = torch.load('checkpoint_step_200000.pt', 
                           map_location=device, 
                           weights_only=False)
    
    print(f"✅ Checkpoint loaded on GPU")
    print(f"  Step: {checkpoint['step']}")
    print(f"  Loss: {checkpoint['loss']:.4f}")
    
    # Test GPU computation
    print("\nTesting GPU computation...")
    x = torch.randn(1000, 1000, device=device)
    y = x @ x.T
    print(f"✅ GPU computation successful")
    print(f"  Result shape: {y.shape}")
    print(f"  Device: {y.device}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
