import sys
import os
sys.path.insert(0, 'c:\\PCOS Filles\\PCOS-Detection\\backend')

import cv2
import numpy as np
from preprocessing.ultrasound_preprocessor import UltrasoundPreprocessor
from configs.preprocessing_config import PreprocessingConfig

# Initialize preprocessor with SRAD config
config = PreprocessingConfig()
config.NOISE_REDUCTION_METHOD = 'srad'
preprocessor = UltrasoundPreprocessor(config)

# Test image
test_image_path = 'c:\\PCOS Filles\\PCOS-Detection\\data\\raw\\UltraSound\\Ovarian_US\\complex_cyst\\complex_cyst_0001.jpg'

print("=" * 70)
print("DETAILED STATISTICAL ANALYSIS BEFORE/AFTER SRAD")
print("=" * 70)

# Load image
img = cv2.imread(test_image_path)

print(f"\nBEFORE SRAD (Original):")
print(f"  Shape: {img.shape}")
print(f"  Dtype: {img.dtype}")
print(f"  Mean: {img.mean():.2f}")
print(f"  Std: {img.std():.2f}")
print(f"  Min: {img.min()}, Max: {img.max()}")
print(f"  Median: {np.median(img):.2f}")
print(f"  25th percentile: {np.percentile(img, 25):.2f}")
print(f"  75th percentile: {np.percentile(img, 75):.2f}")

# Apply SRAD
denoised = preprocessor._reduce_speckle_noise(img)

print(f"\nAFTER SRAD (Denoised):")
print(f"  Shape: {denoised.shape}")
print(f"  Dtype: {denoised.dtype}")
print(f"  Mean: {denoised.mean():.2f}")
print(f"  Std: {denoised.std():.2f}")
print(f"  Min: {denoised.min()}, Max: {denoised.max()}")
print(f"  Median: {np.median(denoised):.2f}")
print(f"  25th percentile: {np.percentile(denoised, 25):.2f}")
print(f"  75th percentile: {np.percentile(denoised, 75):.2f}")

# Calculate noise/texture measure (local variance)
def calculate_local_variance(image, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    local_squared_mean = cv2.filter2D((gray.astype(np.float32))**2, -1, kernel)
    local_variance = local_squared_mean - local_mean**2
    return np.mean(local_variance)

print(f"\nNOISE/TEXTURE MEASURES:")
print(f"  Before SRAD - Mean local variance: {calculate_local_variance(img):.2f}")
print(f"  After SRAD - Mean local variance: {calculate_local_variance(denoised):.2f}")

# Intensity shift analysis
intensity_shift = denoised.mean() - img.mean()
print(f"\nINTENSITY SHIFT ANALYSIS:")
print(f"  Mean intensity shift: {intensity_shift:+.2f} ({(intensity_shift/img.mean()*100):+.2f}%)")

# Check for saturation
before_saturation = (img == 255).sum() / img.size * 100
after_saturation = (denoised == 255).sum() / denoised.size * 100
print(f"\nSATURATION ANALYSIS:")
print(f"  Before SRAD - Saturated pixels: {before_saturation:.2f}%")
print(f"  After SRAD - Saturated pixels: {after_saturation:.2f}%")
