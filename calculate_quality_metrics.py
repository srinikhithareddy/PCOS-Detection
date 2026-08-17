import sys
import os
sys.path.insert(0, 'c:\\PCOS Filles\\PCOS-Detection\\backend')

import cv2
import numpy as np
import torch

try:
    import pyiqa
    print("pyiqa available")
except ImportError:
    print("pyiqa not available, installing...")
    os.system("pip install pyiqa")
    import pyiqa

# Load images
resized_path = 'c:\\PCOS Filles\\PCOS-Detection\\data\\processed\\ultrasound\\resized\\complex_cyst_0001_resized.jpg'
denoised_path = 'c:\\PCOS Filles\\PCOS-Detection\\data\\processed\\ultrasound\\denoised\\complex_cyst_0001_denoised.jpg'

resized_img = cv2.imread(resized_path)
denoised_img = cv2.imread(denoised_path)

# Initialize quality metrics
brisque_metric = pyiqa.create_metric('brisque', device='cpu')
niqe_metric = pyiqa.create_metric('niqe', device='cpu')

# Convert BGR to RGB for pyiqa
resized_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
denoised_rgb = cv2.cvtColor(denoised_img, cv2.COLOR_BGR2RGB)

# Calculate metrics
print("=" * 60)
print("QUALITY METRICS BEFORE/AFTER SRAD")
print("=" * 60)

# BRISQUE (lower is better) - add batch dimension
brisque_before = brisque_metric(torch.from_numpy(resized_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0)
brisque_after = brisque_metric(torch.from_numpy(denoised_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0)

print(f"\nBRISQUE (lower is better):")
print(f"  Before SRAD: {brisque_before.item():.2f}")
print(f"  After SRAD:  {brisque_after.item():.2f}")
print(f"  Change:       {brisque_after.item() - brisque_before.item():.2f}")

# NIQE (lower is better, closer to natural images) - add batch dimension
niqe_before = niqe_metric(torch.from_numpy(resized_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0)
niqe_after = niqe_metric(torch.from_numpy(denoised_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0)

print(f"\nNIQE (lower is better):")
print(f"  Before SRAD: {niqe_before.item():.2f}")
print(f"  After SRAD:  {niqe_after.item():.2f}")
print(f"  Change:       {niqe_after.item() - niqe_before.item():.2f}")

print("\n" + "=" * 60)
print("METRIC INTERPRETATION")
print("=" * 60)
print("BRISQUE: Measures blur/artifacts. Lower = better quality.")
print("NIQE: Measures deviation from natural image statistics. Lower = more natural.")
print()
print("For speckle reduction:")
print("- BRISQUE may increase if smoothing is too aggressive")
print("- NIQE may decrease if noise is reduced while preserving structure")
print("- Optimal: Noise reduction without excessive smoothing")
