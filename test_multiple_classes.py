import sys
import os
sys.path.insert(0, 'c:\\PCOS Filles\\PCOS-Detection\\backend')

import cv2
import numpy as np
import torch
import pyiqa
from preprocessing.ultrasound_preprocessor import UltrasoundPreprocessor
from configs.preprocessing_config import PreprocessingConfig

# Initialize quality metrics
brisque_metric = pyiqa.create_metric('brisque', device='cpu')
niqe_metric = pyiqa.create_metric('niqe', device='cpu')

# Initialize preprocessor with SRAD config
config = PreprocessingConfig()
config.NOISE_REDUCTION_METHOD = 'srad'
preprocessor = UltrasoundPreprocessor(config)

# Test images from each class
test_images = {
    'healthy': 'c:\\PCOS Filles\\PCOS-Detection\\data\\raw\\UltraSound\\Ovarian_US\\healthy\\healthy_0001.jpg',
    'dominant_follicle': 'c:\\PCOS Filles\\PCOS-Detection\\data\\raw\\UltraSound\\Ovarian_US\\dominant_follicle\\dominant_follicle_0001.jpg',
    'poly_cyst': 'c:\\PCOS Filles\\PCOS-Detection\\data\\raw\\UltraSound\\Ovarian_US\\poly_cyst\\poly_cyst_0001.jpg',
    'simple_cyst': 'c:\\PCOS Filles\\PCOS-Detection\\data\\raw\\UltraSound\\Ovarian_US\\simple_cyst\\simple_cyst_0001.jpg',
    'complex_cyst': 'c:\\PCOS Filles\\PCOS-Detection\\data\\raw\\UltraSound\\Ovarian_US\\complex_cyst\\complex_cyst_0001.jpg'
}

print("=" * 80)
print("SRAD TESTING ON MULTIPLE IMAGE CLASSES")
print("=" * 80)
print()
print(f"{'Class':<20} {'BRISQUE Before':<15} {'BRISQUE After':<15} {'NIQE Before':<15} {'NIQE After':<15} {'Visual':<10}")
print("-" * 100)

results = []

for class_name, image_path in test_images.items():
    if not os.path.exists(image_path):
        print(f"{class_name:<20} {'Image not found':<70}")
        continue
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"{class_name:<20} {'Failed to load':<70}")
        continue
    
    # Apply SRAD
    denoised = preprocessor._reduce_speckle_noise(img)
    
    # Convert to RGB for quality metrics
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    denoised_rgb = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)
    
    # Calculate quality metrics
    try:
        brisque_before = brisque_metric(torch.from_numpy(img_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0)
        brisque_after = brisque_metric(torch.from_numpy(denoised_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0)
        niqe_before = niqe_metric(torch.from_numpy(img_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0)
        niqe_after = niqe_metric(torch.from_numpy(denoised_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0)
        
        # Calculate pixel difference
        diff = cv2.absdiff(img, denoised)
        diff_pixels = np.count_nonzero(diff)
        total_pixels = img.size
        diff_percentage = (diff_pixels / total_pixels) * 100
        
        # Visual assessment
        if diff_percentage > 20:
            visual = "Good"
        elif diff_percentage > 5:
            visual = "Moderate"
        else:
            visual = "Minimal"
        
        print(f"{class_name:<20} {brisque_before.item():<15.2f} {brisque_after.item():<15.2f} {niqe_before.item():<15.2f} {niqe_after.item():<15.2f} {visual:<10}")
        
        results.append({
            'class': class_name,
            'brisque_before': brisque_before.item(),
            'brisque_after': brisque_after.item(),
            'niqe_before': niqe_before.item(),
            'niqe_after': niqe_after.item(),
            'diff_percentage': diff_percentage,
            'visual': visual
        })
        
    except Exception as e:
        print(f"{class_name:<20} {'Error: ' + str(e):<70}")

print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)

if results:
    avg_brisque_change = np.mean([r['brisque_after'] - r['brisque_before'] for r in results])
    avg_niqe_change = np.mean([r['niqe_after'] - r['niqe_before'] for r in results])
    avg_diff_percentage = np.mean([r['diff_percentage'] for r in results])
    
    print(f"Average BRISQUE change: {avg_brisque_change:.2f}")
    print(f"Average NIQE change: {avg_niqe_change:.2f}")
    print(f"Average pixel difference: {avg_diff_percentage:.2f}%")
    print()
    
    print("Interpretation:")
    print("- BRISQUE increase suggests smoothing (expected for denoising)")
    print("- NIQE decrease suggests more natural appearance after noise reduction")
    print("- Pixel differences indicate actual processing is occurring")
