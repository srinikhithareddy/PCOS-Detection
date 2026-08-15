"""
Ultrasound Image Quality Assessment Module
Uses BRISQUE and NIQE for no-reference image quality assessment
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config
from pyiqa import create_metric
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns


class QualityAssessment:
    """Quality assessment for ultrasound images using pyiqa BRISQUE and NIQE"""
    
    def __init__(self):
        # Initialize pyiqa metrics
        self.brisque_model = create_metric('brisque', device='cpu')
        self.niqe_model = create_metric('niqe', device='cpu')
        
        # Load V2 thresholds from config
        self.brisque_good_threshold = config.Config.BRISQUE_GOOD_THRESHOLD
        self.brisque_poor_threshold = config.Config.BRISQUE_POOR_THRESHOLD
        self.niqe_good_threshold = config.Config.NIQE_GOOD_THRESHOLD
        self.niqe_poor_threshold = config.Config.NIQE_POOR_THRESHOLD
    
    def calculate_brisque(self, image_path: str) -> float:
        """
        Calculate BRISQUE score using pyiqa
        Lower scores indicate better quality
        
        Args:
            image_path: Path to image file
            
        Returns:
            BRISQUE score as float, or None if error
        """
        try:
            score = self.brisque_model(image_path)
            return float(score.item())
        except Exception as e:
            print(f"Error calculating BRISQUE for {image_path}: {e}")
            return None
    
    def calculate_niqe(self, image_path: str) -> float:
        """
        Calculate NIQE score using pyiqa
        Lower scores indicate better quality
        
        Args:
            image_path: Path to image file
            
        Returns:
            NIQE score as float, or None if error
        """
        try:
            score = self.niqe_model(image_path)
            return float(score.item())
        except Exception as e:
            print(f"Error calculating NIQE for {image_path}: {e}")
            return None
    
    def assess_quality(self, image_path: Path) -> Dict[str, any]:
        """
        Assess quality of a single image using V2 classification logic
        Returns dictionary with individual BRISQUE, NIQE scores, and V2 classification
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with filename, brisque_score, niqe_score, V2 classification, and error (if any)
        """
        try:
            # Calculate individual quality scores using file paths (no image modification)
            brisque_score = self.calculate_brisque(str(image_path))
            niqe_score = self.calculate_niqe(str(image_path))
            
            # V2 Classification Logic
            # BRISQUE classification
            if brisque_score <= self.brisque_good_threshold:
                brisque_cat = 'good'
            elif brisque_score <= self.brisque_poor_threshold:
                brisque_cat = 'poor'
            else:
                brisque_cat = 'unusable'
            
            # NIQE classification
            if niqe_score <= self.niqe_good_threshold:
                niqe_cat = 'good'
            elif niqe_score <= self.niqe_poor_threshold:
                niqe_cat = 'poor'
            else:
                niqe_cat = 'unusable'
            
            # V2 Combined classification logic
            if brisque_cat == 'good' and niqe_cat == 'good':
                quality_category = 'good'
                decision = 'continue'
            elif brisque_cat == 'unusable' and niqe_cat == 'unusable':
                quality_category = 'unusable'
                decision = 'reject'
            elif (brisque_cat == 'unusable' and niqe_cat in ['good', 'poor']) or (niqe_cat == 'unusable' and brisque_cat in ['good', 'poor']):
                quality_category = 'review'
                decision = 'manual_review'
            else:
                quality_category = 'poor'
                decision = 'enhance'
            
            return {
                'filename': image_path.name,
                'brisque_score': brisque_score,
                'niqe_score': niqe_score,
                'brisque_category': brisque_cat,
                'niqe_category': niqe_cat,
                'quality_category': quality_category,
                'preprocessing_decision': decision,
                'error': None
            }
            
        except Exception as e:
            return {
                'filename': image_path.name,
                'brisque_score': None,
                'niqe_score': None,
                'brisque_category': None,
                'niqe_category': None,
                'quality_category': 'unusable',
                'preprocessing_decision': 'reject',
                'error': str(e)
            }


def assess_ultrasound_quality():
    """
    Assess quality of all ultrasound images in the dataset using pyiqa
    Returns DataFrame with individual BRISQUE and NIQE scores
    """
    print("=" * 80)
    print("ULTRASOUND QUALITY ASSESSMENT - pyiqa BRISQUE & NIQE")
    print("=" * 80)
    
    quality_assessor = QualityAssessment()
    
    # Results storage
    all_results = []
    class_counts = {}
    
    # Process each class
    for class_name in config.Config.ULTRASOUND_CLASSES:
        class_path = Path(config.Config.ULTRASOUND_RAW) / class_name
        
        if not class_path.exists():
            print(f"Warning: Class directory not found: {class_path}")
            continue
        
        # Get all image files (exclude desktop.ini and other system files)
        image_files = [f for f in class_path.glob('*.jpg') if f.name != 'desktop.ini']
        
        print(f"\nProcessing class: {class_name}")
        print(f"Found {len(image_files)} images")
        
        class_results = []
        
        for img_file in image_files:
            result = quality_assessor.assess_quality(img_file)
            result['class'] = class_name
            result['filepath'] = str(img_file)
            class_results.append(result)
            all_results.append(result)
        
        class_counts[class_name] = len(class_results)
        print(f"Processed {len(class_results)} images")
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Save report
    report_path = Path(config.Config.REPORTS_DIR) / 'ultrasound_quality_report.csv'
    df.to_csv(report_path, index=False)
    print(f"\nQuality report saved to: {report_path}")
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("QUALITY ASSESSMENT SUMMARY")
    print("=" * 80)
    
    total_images = len(df)
    valid_brisque = df['brisque_score'].notna().sum()
    valid_niqe = df['niqe_score'].notna().sum()
    errors = df['error'].notna().sum()
    
    print(f"\nTotal images processed: {total_images}")
    print(f"Valid BRISQUE scores: {valid_brisque} ({valid_brisque/total_images*100:.2f}%)")
    print(f"Valid NIQE scores: {valid_niqe} ({valid_niqe/total_images*100:.2f}%)")
    print(f"Errors: {errors} ({errors/total_images*100:.2f}%)")
    
    print("\nImages per class:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count}")
    
    # BRISQUE statistics
    if valid_brisque > 0:
        brisque_scores = df['brisque_score'].dropna()
        print(f"\nBRISQUE Statistics:")
        print(f"  Mean: {brisque_scores.mean():.4f}")
        print(f"  Std: {brisque_scores.std():.4f}")
        print(f"  Min: {brisque_scores.min():.4f}")
        print(f"  Max: {brisque_scores.max():.4f}")
        print(f"  Median: {brisque_scores.median():.4f}")
        print(f"  25th percentile: {brisque_scores.quantile(0.25):.4f}")
        print(f"  75th percentile: {brisque_scores.quantile(0.75):.4f}")
    
    # NIQE statistics
    if valid_niqe > 0:
        niqe_scores = df['niqe_score'].dropna()
        print(f"\nNIQE Statistics:")
        print(f"  Mean: {niqe_scores.mean():.4f}")
        print(f"  Std: {niqe_scores.std():.4f}")
        print(f"  Min: {niqe_scores.min():.4f}")
        print(f"  Max: {niqe_scores.max():.4f}")
        print(f"  Median: {niqe_scores.median():.4f}")
        print(f"  25th percentile: {niqe_scores.quantile(0.25):.4f}")
        print(f"  75th percentile: {niqe_scores.quantile(0.75):.4f}")
    
    return df


if __name__ == "__main__":
    df = assess_ultrasound_quality()
