import sys
import os
sys.path.insert(0, 'c:\\PCOS Filles\\PCOS-Detection\\backend')

import cv2
import numpy as np
from preprocessing.ultrasound_preprocessor import UltrasoundPreprocessor
from configs.preprocessing_config import PreprocessingConfig

# Load the resized image
resized_path = 'c:\\PCOS Filles\\PCOS-Detection\\data\\processed\\ultrasound\\resized\\complex_cyst_0001_resized.jpg'
resized_img = cv2.imread(resized_path)

if resized_img is None:
    print("ERROR: Could not load resized image")
    sys.exit(1)

print("=" * 60)
print("SRAD VALIDATION TEST")
print("=" * 60)
print(f"\nInput image: {resized_path}")
print(f"Shape: {resized_img.shape}")
print(f"Dtype: {resized_img.dtype}")
print(f"File size: {os.path.getsize(resized_path)} bytes")

# Get statistics before SRAD
print(f"\nRESIZED Statistics:")
print(f"  Mean: {resized_img.mean():.2f}")
print(f"  Std: {resized_img.std():.2f}")
print(f"  Min: {resized_img.min()}, Max: {resized_img.max()}")

# Initialize preprocessor with SRAD config
config = PreprocessingConfig()
config.NOISE_REDUCTION_METHOD = 'srad'
preprocessor = UltrasoundPreprocessor(config)

# Apply SRAD
print(f"\nApplying SRAD with parameters:")
print(f"  Iterations: {config.SRAD_ITERATIONS}")
print(f"  Time step: {config.SRAD_TIME_STEP}")
print(f"  Q0: {config.SRAD_Q0}")
print(f"  Kernel size: {config.SRAD_KERNEL_SIZE}")

# Convert image bytes to simulate pipeline input
import io
from PIL import Image
pil_img = Image.fromarray(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB))
img_bytes = io.BytesIO()
pil_img.save(img_bytes, format='JPEG')
img_bytes.seek(0)

# Apply SRAD directly using the internal method
denoised_img = preprocessor._reduce_speckle_noise(resized_img)

print(f"\nSRAD DENOISED Statistics:")
print(f"  Shape: {denoised_img.shape}")
print(f"  Dtype: {denoised_img.dtype}")
print(f"  Mean: {denoised_img.mean():.2f}")
print(f"  Std: {denoised_img.std():.2f}")
print(f"  Min: {denoised_img.min()}, Max: {denoised_img.max()}")

# Calculate differences
diff = cv2.absdiff(resized_img, denoised_img)
diff_pixels = np.count_nonzero(diff)
total_pixels = resized_img.size
diff_percentage = (diff_pixels / total_pixels) * 100
mean_diff = np.mean(diff)
max_diff = np.max(diff)

print(f"\nDifference Analysis:")
print(f"  Different pixels: {diff_pixels}/{total_pixels} ({diff_percentage:.2f}%)")
print(f"  Mean absolute difference: {mean_diff:.2f}")
print(f"  Maximum pixel difference: {max_diff}")

if diff_pixels == 0:
    print("  *** STILL IDENTICAL - SRAD NOT WORKING ***")
elif diff_percentage < 0.1:
    print("  *** NEARLY IDENTICAL - SRAD BARELY WORKING ***")
else:
    print("  *** SRAD IS PRODUCING CHANGES ***")

# Save denoised image for comparison
output_path = 'c:\\PCOS Filles\\PCOS-Detection\\data\\processed\\ultrasound\\denoised\\complex_cyst_0001_denoised.jpg'
cv2.imwrite(output_path, denoised_img)
print(f"\nSaved SRAD denoised image to: {output_path}")

# Save difference image
diff_path = 'c:\\PCOS Filles\\PCOS-Detection\\temp_difference.jpg'
cv2.imwrite(diff_path, diff)
print(f"Saved difference image to: {diff_path}")
