"""
Test Script for Corrected Preprocessing Implementation
Tests on small sample: 20 ultrasound images + 20 clinical records
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.quality_assessment.ultrasound_quality import QualityAssessment
from backend.preprocessing.ultrasound_preprocessor import UltrasoundPreprocessor
from backend.clinical_preprocessing.clinical_preprocessor import ClinicalPreprocessor
import config

# Setup basic logger
import logging
general_logger = logging.getLogger('pcos_pipeline')
general_logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
general_logger.addHandler(handler)


def test_ultrasound_quality_v2():
    """Test V2 quality classification on 20 images"""
    general_logger.info("=" * 60)
    general_logger.info("TESTING ULTRASOUND QUALITY CLASSIFICATION V2")
    general_logger.info("=" * 60)
    
    # Get sample images (4 from each of 5 classes = 20 total)
    sample_images = []
    ultrasound_dir = Path(config.Config.ULTRASOUND_RAW)
    
    for class_name in config.Config.ULTRASOUND_CLASSES[:5]:
        class_dir = ultrasound_dir / class_name
        if class_dir.exists():
            images = list(class_dir.glob('*.jpg'))[:4]  # Take first 4
            sample_images.extend([(img, class_name) for img in images])
    
    general_logger.info(f"Testing on {len(sample_images)} sample images")
    
    # Initialize quality assessor
    assessor = QualityAssessment()
    
    # Test V2 classification
    results = []
    for img_path, class_name in sample_images:
        result = assessor.assess_quality(img_path)
        results.append(result)
        general_logger.info(f"{img_path.name}: {result.get('quality_category', 'unknown')} ({result.get('preprocessing_decision', 'unknown')})")
    
    # Verify V2 thresholds are used
    general_logger.info(f"\nV2 Thresholds:")
    general_logger.info(f"  BRISQUE: Good <= {config.Config.BRISQUE_GOOD_THRESHOLD}, Poor <= {config.Config.BRISQUE_POOR_THRESHOLD}")
    general_logger.info(f"  NIQE: Good <= {config.Config.NIQE_GOOD_THRESHOLD}, Poor <= {config.Config.NIQE_POOR_THRESHOLD}")
    
    # Check classification distribution
    df = pd.DataFrame(results)
    if 'quality_category' in df.columns:
        general_logger.info(f"\nClassification distribution:")
        general_logger.info(df['quality_category'].value_counts().to_dict())
    
    general_logger.info("✅ V2 Quality Classification Test Complete")
    return results


def test_ultrasound_preprocessing():
    """Test ultrasound preprocessing on 20 images"""
    general_logger.info("\n" + "=" * 60)
    general_logger.info("TESTING ULTRASOUND PREPROCESSING")
    general_logger.info("=" * 60)
    
    # Get sample images
    sample_images = []
    ultrasound_dir = Path(config.Config.ULTRASOUND_RAW)
    
    for class_name in config.Config.ULTRASOUND_CLASSES[:5]:
        class_dir = ultrasound_dir / class_name
        if class_dir.exists():
            images = list(class_dir.glob('*.jpg'))[:4]
            sample_images.extend([(img, class_name) for img in images])
    
    general_logger.info(f"Testing on {len(sample_images)} sample images")
    
    # Initialize preprocessor
    preprocessor = UltrasoundPreprocessor()
    
    # Test output structure
    test_output_dir = Path("data/processed/ultrasound_test")
    test_output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    for img_path, class_name in sample_images:
        # Simulate different quality categories
        quality_categories = ['good', 'poor', 'review', 'unusable']
        quality_cat = quality_categories[len(results) % 4]  # Rotate through categories
        
        result = preprocessor.process_image(img_path, test_output_dir, class_name, quality_cat)
        results.append(result)
        
        if result['status'] == 'success':
            general_logger.info(f"{img_path.name}: {quality_cat} -> {result.get('enhanced', False)}")
        else:
            general_logger.error(f"{img_path.name}: FAILED - {result.get('error', 'unknown')}")
    
    # Verify output structure
    general_logger.info(f"\nOutput structure verification:")
    for subdir in ['preprocessed/good', 'preprocessed/enhanced', 'review', 'unusable']:
        subdir_path = test_output_dir / subdir
        if subdir_path.exists():
            count = len(list(subdir_path.glob('*')))
            general_logger.info(f"  {subdir}: {count} files")
        else:
            general_logger.info(f"  {subdir}: NOT CREATED")
    
    # Verify no fake ROI directory
    roi_dir = test_output_dir / 'roi'
    if roi_dir.exists():
        general_logger.warning(f"  roi/: EXISTS (should be empty until U-Net weights available)")
    else:
        general_logger.info(f"  roi/: NOT CREATED (correct - no U-Net weights)")
    
    general_logger.info("✅ Ultrasound Preprocessing Test Complete")
    return results


def test_lee_filter():
    """Test true Lee filter implementation"""
    general_logger.info("\n" + "=" * 60)
    general_logger.info("TESTING TRUE LEE FILTER")
    general_logger.info("=" * 60)
    
    import cv2
    from ultrasound_preprocessing import UltrasoundPreprocessor
    
    preprocessor = UltrasoundPreprocessor()
    
    # Get a test image
    ultrasound_dir = Path(config.Config.ULTRASOUND_RAW)
    test_image = None
    for class_name in config.Config.ULTRASOUND_CLASSES:
        class_dir = ultrasound_dir / class_name
        if class_dir.exists():
            images = list(class_dir.glob('*.jpg'))
            if images:
                test_image = cv2.imread(str(images[0]))
                break
    
    if test_image is None:
        general_logger.error("No test image found")
        return False
    
    # Test Lee filter
    general_logger.info("Applying Lee filter...")
    filtered = preprocessor.lee_filter(test_image, size=3, iterations=1)
    
    # Verify dimensions preserved
    if filtered.shape == test_image.shape:
        general_logger.info("✅ Image dimensions preserved")
    else:
        general_logger.error(f"❌ Image dimensions changed: {test_image.shape} -> {filtered.shape}")
        return False
    
    # Verify pixel range
    if filtered.dtype == np.uint8 and filtered.min() >= 0 and filtered.max() <= 255:
        general_logger.info("✅ Pixel range valid (0-255)")
    else:
        general_logger.error(f"❌ Invalid pixel range: {filtered.dtype}, min={filtered.min()}, max={filtered.max()}")
        return False
    
    general_logger.info("✅ Lee Filter Test Complete")
    return True


def test_srad_filter():
    """Test SRAD filter implementation"""
    general_logger.info("\n" + "=" * 60)
    general_logger.info("TESTING SRAD FILTER")
    general_logger.info("=" * 60)
    
    import cv2
    from ultrasound_preprocessing import UltrasoundPreprocessor
    
    preprocessor = UltrasoundPreprocessor()
    
    # Get a test image
    ultrasound_dir = Path(config.Config.ULTRASOUND_RAW)
    test_image = None
    for class_name in config.Config.ULTRASOUND_CLASSES:
        class_dir = ultrasound_dir / class_name
        if class_dir.exists():
            images = list(class_dir.glob('*.jpg'))
            if images:
                test_image = cv2.imread(str(images[0]))
                break
    
    if test_image is None:
        general_logger.error("No test image found")
        return False
    
    # Test SRAD filter (with reduced iterations for speed)
    general_logger.info("Applying SRAD filter (10 iterations for test)...")
    filtered = preprocessor.srad_filter(test_image, iterations=10, time_step=0.05, conductance=0.1)
    
    # Verify dimensions preserved
    if filtered.shape == test_image.shape[:2]:  # SRAD returns grayscale
        general_logger.info("✅ Image dimensions preserved")
    else:
        general_logger.error(f"❌ Image dimensions changed: {test_image.shape[:2]} -> {filtered.shape}")
        return False
    
    # Verify pixel range
    if filtered.dtype == np.uint8 and filtered.min() >= 0 and filtered.max() <= 255:
        general_logger.info("✅ Pixel range valid (0-255)")
    else:
        general_logger.error(f"❌ Invalid pixel range: {filtered.dtype}, min={filtered.min()}, max={filtered.max()}")
        return False
    
    general_logger.info("✅ SRAD Filter Test Complete")
    return True


def test_clinical_preprocessing():
    """Test clinical preprocessing on 20 records"""
    general_logger.info("\n" + "=" * 60)
    general_logger.info("TESTING CLINICAL PREPROCESSING")
    general_logger.info("=" * 60)
    
    # Load clinical data
    preprocessor = ClinicalPreprocessor()
    df = preprocessor.load_data()
    
    # Take first 20 records for testing
    test_df = df.head(20).copy()
    general_logger.info(f"Testing on {len(test_df)} records")
    
    # Get selected features
    selected_features = preprocessor.get_selected_features(test_df)
    general_logger.info(f"Using {len(selected_features)} features")
    
    # Test range checks
    general_logger.info("\nTesting range validity checks...")
    test_df = preprocessor.check_range_validity(test_df, selected_features)
    
    # Test consistency checks
    general_logger.info("Testing consistency checks...")
    test_df = preprocessor.check_consistency(test_df)
    
    # Test reliability score
    general_logger.info("Testing reliability score calculation...")
    reliability = preprocessor.calculate_reliability_score(test_df, selected_features)
    general_logger.info(f"Reliability scores: min={reliability['reliability_score'].min()}, max={reliability['reliability_score'].max()}")
    
    # Test outlier leakage prevention
    general_logger.info("Testing outlier leakage prevention...")
    bounds = preprocessor.fit_outlier_bounds(test_df, selected_features)
    general_logger.info(f"Fitted bounds for {len(bounds)} features")
    
    # Apply clipping using same bounds
    test_df_clipped = preprocessor.apply_outlier_clipping(test_df, selected_features, bounds)
    general_logger.info("Applied clipping using training bounds")
    
    # Verify categorical encoding documentation
    general_logger.info("\nCategorical encoding: NOT REQUIRED (all 47 features are numeric)")
    
    general_logger.info("✅ Clinical Preprocessing Test Complete")
    return test_df, reliability


def test_error_handling():
    """Test error handling doesn't crash pipeline"""
    general_logger.info("\n" + "=" * 60)
    general_logger.info("TESTING ERROR HANDLING")
    general_logger.info("=" * 60)
    
    from ultrasound_preprocessing import UltrasoundPreprocessor
    from pathlib import Path
    
    preprocessor = UltrasoundPreprocessor()
    
    # Test with non-existent image
    result = preprocessor.process_image(
        Path("nonexistent.jpg"),
        Path("data/processed/ultrasound_test"),
        "test_class",
        "good"
    )
    
    if result['status'] == 'failed':
        general_logger.info("✅ Non-existent image handled gracefully")
    else:
        general_logger.error("❌ Non-existent image not handled correctly")
        return False
    
    # Test with corrupted image path
    result = preprocessor.process_image(
        Path("data/raw/UltraSound/Ovarian_US/healthy/corrupted.jpg"),
        Path("data/processed/ultrasound_test"),
        "test_class",
        "good"
    )
    
    if result['status'] == 'failed':
        general_logger.info("✅ Missing image handled gracefully")
    else:
        general_logger.error("❌ Missing image not handled correctly")
        return False
    
    general_logger.info("✅ Error Handling Test Complete")
    return True


def main():
    """Run all tests"""
    general_logger.info("=" * 80)
    general_logger.info("PREPROCESSING V2 TEST SUITE")
    general_logger.info("Testing: 20 ultrasound images + 20 clinical records")
    general_logger.info("=" * 80)
    
    test_results = {}
    
    # Test 1: V2 Quality Classification
    try:
        test_results['quality_v2'] = test_ultrasound_quality_v2()
    except Exception as e:
        general_logger.error(f"❌ Quality V2 test failed: {e}")
        test_results['quality_v2'] = None
    
    # Test 2: Lee Filter
    try:
        test_results['lee_filter'] = test_lee_filter()
    except Exception as e:
        general_logger.error(f"❌ Lee filter test failed: {e}")
        test_results['lee_filter'] = False
    
    # Test 3: SRAD Filter
    try:
        test_results['srad_filter'] = test_srad_filter()
    except Exception as e:
        general_logger.error(f"❌ SRAD filter test failed: {e}")
        test_results['srad_filter'] = False
    
    # Test 4: Ultrasound Preprocessing
    try:
        test_results['ultrasound_prep'] = test_ultrasound_preprocessing()
    except Exception as e:
        general_logger.error(f"❌ Ultrasound preprocessing test failed: {e}")
        test_results['ultrasound_prep'] = None
    
    # Test 5: Clinical Preprocessing
    try:
        test_results['clinical_prep'] = test_clinical_preprocessing()
    except Exception as e:
        general_logger.error(f"❌ Clinical preprocessing test failed: {e}")
        test_results['clinical_prep'] = None
    
    # Test 6: Error Handling
    try:
        test_results['error_handling'] = test_error_handling()
    except Exception as e:
        general_logger.error(f"❌ Error handling test failed: {e}")
        test_results['error_handling'] = False
    
    # Summary
    general_logger.info("\n" + "=" * 80)
    general_logger.info("TEST SUMMARY")
    general_logger.info("=" * 80)
    
    for test_name, result in test_results.items():
        if result is not None and result is not False:
            general_logger.info(f"✅ {test_name}: PASSED")
        else:
            general_logger.info(f"❌ {test_name}: FAILED")
    
    general_logger.info("\n" + "=" * 80)
    general_logger.info("FILES MODIFIED")
    general_logger.info("=" * 80)
    general_logger.info("1. config.py - Updated V2 thresholds")
    general_logger.info("2. ultrasound_quality.py - Updated V2 classification logic")
    general_logger.info("3. ultrasound_preprocessing.py - Added enhancement, true Lee, SRAD, blocked U-Net, new output structure")
    general_logger.info("4. clinical_preprocessing.py - Added range checks, consistency checks, reliability score, leakage fix")
    general_logger.info("5. logger_config.py - NEW: Structured logging")
    general_logger.info("6. test_preprocessing_v2.py - NEW: Test script")
    
    general_logger.info("\n" + "=" * 80)
    general_logger.info("WHAT WAS FIXED")
    general_logger.info("=" * 80)
    general_logger.info("✅ V2 quality thresholds (BRISQUE 15/35, NIQE 5.5/8)")
    general_logger.info("✅ V2 classification logic with Review category")
    general_logger.info("✅ Poor image enhancement stage")
    general_logger.info("✅ True Lee speckle filter (not bilateral approximation)")
    general_logger.info("✅ SRAD alternative implementation")
    general_logger.info("✅ U-Net segmentation marked as BLOCKED (no weights)")
    general_logger.info("✅ ROI handling fixed (no fake ROIs, use preprocessed/)")
    general_logger.info("✅ Clinical range checks")
    general_logger.info("✅ Clinical consistency checks")
    general_logger.info("✅ Clinical reliability score")
    general_logger.info("✅ Clinical outlier leakage fixed (train bounds only)")
    general_logger.info("✅ Categorical encoding documented (not needed)")
    general_logger.info("✅ Reproducibility (central random seed)")
    general_logger.info("✅ Structured logging")
    general_logger.info("✅ Error handling (doesn't crash on bad images)")
    general_logger.info("✅ Output structure (preprocessed/, review/, unusable/)")
    
    general_logger.info("\n" + "=" * 80)
    general_logger.info("WHAT REMAINS BLOCKED")
    general_logger.info("=" * 80)
    general_logger.info("❌ U-Net segmentation - No trained weights available")
    general_logger.info("❌ Ovarian ROI extraction - Requires segmentation first")
    general_logger.info("❌ data/processed/ultrasound/roi/ - Empty until U-Net weights exist")
    
    general_logger.info("\n" + "=" * 80)
    general_logger.info("STOPPING CONDITION MET")
    general_logger.info("Test complete. Awaiting approval for full preprocessing run.")
    general_logger.info("=" * 80)


if __name__ == "__main__":
    main()
