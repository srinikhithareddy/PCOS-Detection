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
print("EDGE PRESERVATION ANALYSIS")
print("=" * 70)

# Load image
img = cv2.imread(test_image_path)
if len(img.shape) == 3:
    gray_original = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
else:
    gray_original = img.copy()

# Apply SRAD
denoised = preprocessor._reduce_speckle_noise(img)
if len(denoised.shape) == 3:
    gray_denoised = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
else:
    gray_denoised = denoised.copy()

# Sobel edge magnitude
sobel_x_orig = cv2.Sobel(gray_original, cv2.CV_64F, 1, 0, ksize=3)
sobel_y_orig = cv2.Sobel(gray_original, cv2.CV_64F, 0, 1, ksize=3)
edge_magnitude_orig = np.sqrt(sobel_x_orig**2 + sobel_y_orig**2)

sobel_x_denoised = cv2.Sobel(gray_denoised, cv2.CV_64F, 1, 0, ksize=3)
sobel_y_denoised = cv2.Sobel(gray_denoised, cv2.CV_64F, 0, 1, ksize=3)
edge_magnitude_denoised = np.sqrt(sobel_x_denoised**2 + sobel_y_denoised**2)

print(f"\nSOBEL EDGE MAGNITUDE STATISTICS:")
print(f"  Before SRAD - Mean: {edge_magnitude_orig.mean():.2f}, Std: {edge_magnitude_orig.std():.2f}")
print(f"  After SRAD - Mean: {edge_magnitude_denoised.mean():.2f}, Std: {edge_magnitude_denoised.std():.2f}")

# Canny edge detection
edges_orig = cv2.Canny(gray_original, 50, 150)
edges_denoised = cv2.Canny(gray_denoised, 50, 150)

edge_count_orig = np.sum(edges_orig > 0)
edge_count_denoised = np.sum(edges_denoised > 0)

print(f"\nCANNY EDGE COUNT:")
print(f"  Before SRAD: {edge_count_orig} edge pixels")
print(f"  After SRAD: {edge_count_denoised} edge pixels")
print(f"  Edge preservation ratio: {(edge_count_denoised/edge_count_orig)*100:.2f}%")

# Edge preservation index (EPI)
# EPI = sum(|edge_denoised|) / sum(|edge_original|)
epi = np.sum(np.abs(edge_magnitude_denoised)) / np.sum(np.abs(edge_magnitude_orig))
print(f"\nEDGE PRESERVATION INDEX (EPI):")
print(f"  EPI = {epi:.4f}")
print(f"  (EPI > 1.0 indicates edge enhancement, < 1.0 indicates edge reduction)")

# Noise reduction in non-edge regions
# Create edge mask
edge_mask = edges_orig > 0
non_edge_mask = ~edge_mask

noise_orig = gray_original[non_edge_mask].std()
noise_denoised = gray_denoised[non_edge_mask].std()

print(f"\nNOISE IN NON-EDGE REGIONS:")
print(f"  Before SRAD: {noise_orig:.2f}")
print(f"  After SRAD: {noise_denoised:.2f}")
print(f"  Noise reduction: {(1 - noise_denoised/noise_orig)*100:.2f}%")

# Gradient statistics
grad_x_orig = cv2.Sobel(gray_original, cv2.CV_64F, 1, 0)
grad_y_orig = cv2.Sobel(gray_original, cv2.CV_64F, 0, 1)
grad_magnitude_orig = np.sqrt(grad_x_orig**2 + grad_y_orig**2)

grad_x_denoised = cv2.Sobel(gray_denoised, cv2.CV_64F, 1, 0)
grad_y_denoised = cv2.Sobel(gray_denoised, cv2.CV_64F, 0, 1)
grad_magnitude_denoised = np.sqrt(grad_x_denoised**2 + grad_y_denoised**2)

print(f"\nGRADIENT MAGNITUDE STATISTICS:")
print(f"  Before SRAD - Mean: {grad_magnitude_orig.mean():.2f}, Max: {grad_magnitude_orig.max():.2f}")
print(f"  After SRAD - Mean: {grad_magnitude_denoised.mean():.2f}, Max: {grad_magnitude_denoised.max():.2f}")
print(f"  Gradient preservation: {(grad_magnitude_denoised.mean()/grad_magnitude_orig.mean())*100:.2f}%")
