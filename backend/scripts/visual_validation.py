"""
Visual Validation of Ultrasound Quality Assessment
Generate contact sheets for visual inspection
"""

import pandas as pd
import numpy as np
import cv2
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random

print("=" * 80)
print("ULTRASOUND QUALITY VISUAL VALIDATION")
print("=" * 80)

# Load quality report
report_path = Path('data/reports/ultrasound_quality_report.csv')
df = pd.read_csv(report_path)

print(f"\nLoaded quality report: {len(df)} images")

# Create output directory
output_dir = Path('data/reports/visual_validation')
output_dir.mkdir(parents=True, exist_ok=True)

def select_representative_samples(df, category, n_samples, ensure_classes=True):
    """Select representative samples from a category"""
    category_df = df[df['quality_category'] == category].copy()
    
    if len(category_df) == 0:
        print(f"Warning: No images in category '{category}'")
        return []
    
    if ensure_classes:
        # Ensure all classes are represented
        samples = []
        classes = df['class'].unique()
        samples_per_class = max(1, n_samples // len(classes))
        
        for class_name in classes:
            class_samples = category_df[category_df['class'] == class_name]
            if len(class_samples) > 0:
                n = min(samples_per_class, len(class_samples))
                selected = class_samples.sample(n=n, random_state=42)
                samples.append(selected)
        
        samples_df = pd.concat(samples, ignore_index=True)
        
        # If we need more samples, add randomly
        if len(samples_df) < n_samples:
            remaining = n_samples - len(samples_df)
            additional = category_df[~category_df.index.isin(samples_df.index)].sample(
                n=min(remaining, len(category_df) - len(samples_df)), 
                random_state=42
            )
            samples_df = pd.concat([samples_df, additional], ignore_index=True)
        
        return samples_df.head(n_samples)
    else:
        return category_df.sample(n=min(n_samples, len(category_df)), random_state=42)

def select_boundary_samples(df, metric, threshold, tolerance=0.5, n_samples=10):
    """Select samples near a threshold boundary"""
    df_sorted = df.sort_values(by=metric)
    
    # Find samples within tolerance of threshold
    boundary_df = df[(df[metric] >= threshold - tolerance) & (df[metric] <= threshold + tolerance)]
    
    if len(boundary_df) == 0:
        # Expand tolerance
        boundary_df = df[(df[metric] >= threshold - 2*tolerance) & (df[metric] <= threshold + 2*tolerance)]
    
    if len(boundary_df) == 0:
        print(f"Warning: No samples near {metric} threshold {threshold}")
        return []
    
    return boundary_df.sample(n=min(n_samples, len(boundary_df)), random_state=42)

def create_contact_sheet(images_data, output_path, title, cols=5):
    """Create a contact sheet with images and metadata"""
    n_images = len(images_data)
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (_, row) in enumerate(images_data.iterrows()):
        row_idx = idx // cols
        col_idx = idx % cols
        
        # Load image
        img_path = Path(row['filepath'])
        if img_path.exists():
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[row_idx, col_idx].imshow(img, cmap='gray')
        else:
            axes[row_idx, col_idx].text(0.5, 0.5, 'Image not found', 
                                       ha='center', va='center', transform=axes[row_idx, col_idx].transAxes)
        
        axes[row_idx, col_idx].axis('off')
        
        # Add metadata
        metadata = f"{row['filename']}\n"
        metadata += f"Class: {row['class']}\n"
        metadata += f"BRISQUE: {row['brisque_score']:.2f}\n"
        metadata += f"NIQE: {row['niqe_score']:.2f}\n"
        metadata += f"Final: {row['quality_category']}"
        
        axes[row_idx, col_idx].set_title(metadata, fontsize=8, pad=2)
    
    # Remove empty subplots
    for idx in range(n_images, rows * cols):
        row_idx = idx // cols
        col_idx = idx % cols
        fig.delaxes(axes[row_idx, col_idx])
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

# Select samples
print("\n" + "=" * 80)
print("SELECTING REPRESENTATIVE SAMPLES")
print("=" * 80)

# Good samples (at least 10)
print("\nSelecting Good samples...")
good_samples = select_representative_samples(df, 'good', n_samples=15, ensure_classes=True)
print(f"Selected {len(good_samples)} Good samples")

# Poor samples (at least 20)
print("\nSelecting Poor samples...")
poor_samples = select_representative_samples(df, 'poor', n_samples=25, ensure_classes=True)
print(f"Selected {len(poor_samples)} Poor samples")

# Unusable samples (at least 20)
print("\nSelecting Unusable samples...")
unusable_samples = select_representative_samples(df, 'unusable', n_samples=25, ensure_classes=True)
print(f"Selected {len(unusable_samples)} Unusable samples")

# Boundary samples
print("\nSelecting boundary samples...")
boundary_samples_list = []

# BRISQUE near 10.99 (Good/Poor boundary)
brisque_low_boundary = select_boundary_samples(df, 'brisque_score', 10.99, tolerance=1.0, n_samples=5)
if len(brisque_low_boundary) > 0:
    brisque_low_boundary['boundary_type'] = 'BRISQUE ≈ 10.99 (Good/Poor)'
    boundary_samples_list.append(brisque_low_boundary)

# BRISQUE near 27.86 (Poor/Unusable boundary)
brisque_high_boundary = select_boundary_samples(df, 'brisque_score', 27.86, tolerance=1.0, n_samples=5)
if len(brisque_high_boundary) > 0:
    brisque_high_boundary['boundary_type'] = 'BRISQUE ≈ 27.86 (Poor/Unusable)'
    boundary_samples_list.append(brisque_high_boundary)

# NIQE near 4.81 (Good/Poor boundary)
niqe_low_boundary = select_boundary_samples(df, 'niqe_score', 4.81, tolerance=0.3, n_samples=5)
if len(niqe_low_boundary) > 0:
    niqe_low_boundary['boundary_type'] = 'NIQE ≈ 4.81 (Good/Poor)'
    boundary_samples_list.append(niqe_low_boundary)

# NIQE near 6.79 (Poor/Unusable boundary)
niqe_high_boundary = select_boundary_samples(df, 'niqe_score', 6.79, tolerance=0.3, n_samples=5)
if len(niqe_high_boundary) > 0:
    niqe_high_boundary['boundary_type'] = 'NIQE ≈ 6.79 (Poor/Unusable)'
    boundary_samples_list.append(niqe_high_boundary)

if boundary_samples_list:
    boundary_samples = pd.concat(boundary_samples_list, ignore_index=True)
    print(f"Selected {len(boundary_samples)} boundary samples")
else:
    boundary_samples = pd.DataFrame()
    print("No boundary samples selected")

# Generate contact sheets
print("\n" + "=" * 80)
print("GENERATING CONTACT SHEETS")
print("=" * 80)

if len(good_samples) > 0:
    create_contact_sheet(good_samples, output_dir / 'good_samples.png', 
                        'Good Quality Samples (Continue)', cols=5)

if len(poor_samples) > 0:
    create_contact_sheet(poor_samples, output_dir / 'poor_samples.png', 
                        'Poor Quality Samples (Enhance)', cols=5)

if len(unusable_samples) > 0:
    create_contact_sheet(unusable_samples, output_dir / 'unusable_samples.png', 
                        'Unusable Quality Samples (Reject)', cols=5)

if len(boundary_samples) > 0:
    create_contact_sheet(boundary_samples, output_dir / 'boundary_samples.png', 
                        'Boundary Samples (Near Thresholds)', cols=5)

# Save sample metadata
print("\n" + "=" * 80)
print("SAVING SAMPLE METADATA")
print("=" * 80)

good_samples.to_csv(output_dir / 'good_samples_metadata.csv', index=False)
poor_samples.to_csv(output_dir / 'poor_samples_metadata.csv', index=False)
unusable_samples.to_csv(output_dir / 'unusable_samples_metadata.csv', index=False)
if len(boundary_samples) > 0:
    boundary_samples.to_csv(output_dir / 'boundary_samples_metadata.csv', index=False)

print(f"\nSaved metadata files to: {output_dir}")

# Summary
print("\n" + "=" * 80)
print("VISUAL VALIDATION SUMMARY")
print("=" * 80)

print(f"\nGood samples: {len(good_samples)}")
print(f"  Classes represented: {good_samples['class'].nunique()}")
print(f"  Class distribution:")
print(good_samples['class'].value_counts())

print(f"\nPoor samples: {len(poor_samples)}")
print(f"  Classes represented: {poor_samples['class'].nunique()}")
print(f"  Class distribution:")
print(poor_samples['class'].value_counts())

print(f"\nUnusable samples: {len(unusable_samples)}")
print(f"  Classes represented: {unusable_samples['class'].nunique()}")
print(f"  Class distribution:")
print(unusable_samples['class'].value_counts())

if len(boundary_samples) > 0:
    print(f"\nBoundary samples: {len(boundary_samples)}")
    print(f"  Boundary types:")
    print(boundary_samples['boundary_type'].value_counts())

print("\n" + "=" * 80)
print("VISUAL VALIDATION COMPLETE")
print("=" * 80)
print(f"\nContact sheets saved to: {output_dir}")
print(f"Metadata saved to: {output_dir}")
print("\nPlease visually inspect the contact sheets and provide feedback.")
