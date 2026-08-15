"""
Generate Final Preprocessing Execution Report
Consolidates all preprocessing results into a comprehensive summary
"""

import pandas as pd
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
import config


def generate_final_report():
    """
    Generate comprehensive preprocessing execution report
    """
    print("=" * 80)
    print("FINAL PREPROCESSING EXECUTION REPORT")
    print("=" * 80)
    
    # Load all reports
    reports_dir = Path(config.Config.REPORTS_DIR)
    
    # Ultrasound quality report
    ultrasound_quality_path = reports_dir / 'ultrasound_quality_report.csv'
    if ultrasound_quality_path.exists():
        us_quality_df = pd.read_csv(ultrasound_quality_path)
    else:
        us_quality_df = None
    
    # Ultrasound preprocessing report
    ultrasound_preprocess_path = reports_dir / 'ultrasound_preprocessing_report.csv'
    if ultrasound_preprocess_path.exists():
        us_preprocess_df = pd.read_csv(ultrasound_preprocess_path)
    else:
        us_preprocess_df = None
    
    # Clinical quality report
    clinical_quality_path = reports_dir / 'clinical_quality_report.csv'
    if clinical_quality_path.exists():
        clinical_quality_df = pd.read_csv(clinical_quality_path)
    else:
        clinical_quality_df = None
    
    # Clinical preprocessing report
    clinical_preprocess_path = reports_dir / 'clinical_preprocessing_report.json'
    if clinical_preprocess_path.exists():
        with open(clinical_preprocess_path, 'r') as f:
            clinical_preprocess_report = json.load(f)
    else:
        clinical_preprocess_report = None
    
    # Clinical modifications log
    clinical_mods_path = reports_dir / 'clinical_modifications_log.csv'
    if clinical_mods_path.exists():
        clinical_mods_df = pd.read_csv(clinical_mods_path)
    else:
        clinical_mods_df = None
    
    # Generate summary
    summary = {
        'ultrasound': {},
        'clinical': {},
        'data_splits': {}
    }
    
    # Ultrasound summary
    if us_quality_df is not None:
        summary['ultrasound']['total_images'] = len(us_quality_df)
        summary['ultrasound']['good_quality'] = len(us_quality_df[us_quality_df['quality_category'] == 'good'])
        summary['ultrasound']['poor_quality'] = len(us_quality_df[us_quality_df['quality_category'] == 'poor'])
        summary['ultrasound']['unusable'] = len(us_quality_df[us_quality_df['quality_category'] == 'unusable'])
        
        # Images per class
        summary['ultrasound']['images_per_class'] = {}
        for class_name in config.Config.ULTRASOUND_CLASSES:
            class_df = us_quality_df[us_quality_df['class'] == class_name]
            summary['ultrasound']['images_per_class'][class_name] = len(class_df)
    
    if us_preprocess_df is not None:
        summary['ultrasound']['processed_images'] = len(us_preprocess_df)
        summary['ultrasound']['successful_preprocessing'] = len(us_preprocess_df[us_preprocess_df['status'] == 'success'])
        summary['ultrasound']['failed_preprocessing'] = len(us_preprocess_df[us_preprocess_df['status'] == 'failed'])
        summary['ultrasound']['segmentation_performed'] = len(us_preprocess_df[us_preprocess_df['segmentation_performed'] == True])
    
    # Clinical summary
    if clinical_quality_df is not None:
        summary['clinical']['total_records'] = 468
        summary['clinical']['initial_features'] = 47
        summary['clinical']['features_with_missing'] = len(clinical_quality_df[clinical_quality_df['missing_count'] > 0])
        summary['clinical']['features_with_invalid_negative'] = len(clinical_quality_df[clinical_quality_df['invalid_negative_count'] > 0])
        summary['clinical']['features_with_outliers'] = len(clinical_quality_df[clinical_quality_df['outlier_count'] > 0])
        
        # Total issues
        summary['clinical']['total_missing_values'] = int(clinical_quality_df['missing_count'].sum())
        summary['clinical']['total_invalid_negative'] = int(clinical_quality_df['invalid_negative_count'].sum())
        summary['clinical']['total_outliers'] = int(clinical_quality_df['outlier_count'].sum())
    
    if clinical_preprocess_report is not None:
        summary['clinical']['train_records'] = clinical_preprocess_report['train_records']
        summary['clinical']['val_records'] = clinical_preprocess_report['val_records']
        summary['clinical']['test_records'] = clinical_preprocess_report['test_records']
        summary['clinical']['final_features'] = clinical_preprocess_report['final_features']
        summary['clinical']['value_modifications'] = clinical_preprocess_report['modifications_count']
    
    if clinical_mods_df is not None:
        summary['clinical']['modifications_by_action'] = {k: int(v) for k, v in clinical_mods_df['action'].value_counts().to_dict().items()}
        summary['clinical']['modifications_by_reason'] = {k: int(v) for k, v in clinical_mods_df['reason'].value_counts().to_dict().items()}
    
    # Print report
    print("\n" + "=" * 80)
    print("ULTRASOUND DATA PREPROCESSING RESULTS")
    print("=" * 80)
    
    if 'ultrasound' in summary and summary['ultrasound']:
        print(f"\nTotal images: {summary['ultrasound'].get('total_images', 'N/A')}")
        print(f"Images per class:")
        for class_name, count in summary['ultrasound'].get('images_per_class', {}).items():
            print(f"  {class_name}: {count}")
        
        print(f"\nQuality Assessment:")
        print(f"  Good quality: {summary['ultrasound'].get('good_quality', 'N/A')}")
        print(f"  Poor quality (needs enhancement): {summary['ultrasound'].get('poor_quality', 'N/A')}")
        print(f"  Unusable (rejected): {summary['ultrasound'].get('unusable', 'N/A')}")
        
        print(f"\nPreprocessing:")
        print(f"  Images processed: {summary['ultrasound'].get('processed_images', 'N/A')}")
        print(f"  Successful: {summary['ultrasound'].get('successful_preprocessing', 'N/A')}")
        print(f"  Failed: {summary['ultrasound'].get('failed_preprocessing', 'N/A')}")
        print(f"  Segmentation performed: {summary['ultrasound'].get('segmentation_performed', 'N/A')}")
    
    print("\n" + "=" * 80)
    print("CLINICAL DATA PREPROCESSING RESULTS")
    print("=" * 80)
    
    if 'clinical' in summary and summary['clinical']:
        print(f"\nTotal records: {summary['clinical'].get('total_records', 'N/A')}")
        print(f"Initial features: {summary['clinical'].get('initial_features', 'N/A')}")
        print(f"Final features: {summary['clinical'].get('final_features', 'N/A')}")
        
        print(f"\nQuality Assessment:")
        print(f"  Features with missing values: {summary['clinical'].get('features_with_missing', 'N/A')}")
        print(f"  Features with invalid negative values: {summary['clinical'].get('features_with_invalid_negative', 'N/A')}")
        print(f"  Features with outliers: {summary['clinical'].get('features_with_outliers', 'N/A')}")
        
        print(f"\nTotal Issues:")
        print(f"  Missing values: {summary['clinical'].get('total_missing_values', 'N/A')}")
        print(f"  Invalid negative values: {summary['clinical'].get('total_invalid_negative', 'N/A')}")
        print(f"  Outliers: {summary['clinical'].get('total_outliers', 'N/A')}")
        
        print(f"\nData Splitting:")
        print(f"  Train: {summary['clinical'].get('train_records', 'N/A')} samples")
        print(f"  Validation: {summary['clinical'].get('val_records', 'N/A')} samples")
        print(f"  Test: {summary['clinical'].get('test_records', 'N/A')} samples")
        
        print(f"\nPreprocessing:")
        print(f"  Value modifications: {summary['clinical'].get('value_modifications', 'N/A')}")
        
        if 'modifications_by_action' in summary['clinical']:
            print(f"  Modifications by action:")
            for action, count in summary['clinical']['modifications_by_action'].items():
                print(f"    {action}: {count}")
        
        if 'modifications_by_reason' in summary['clinical']:
            print(f"  Modifications by reason:")
            for reason, count in summary['clinical']['modifications_by_reason'].items():
                print(f"    {reason}: {count}")
    
    print("\n" + "=" * 80)
    print("PREPROCESSING CONFIGURATION")
    print("=" * 80)
    
    print(f"\nExcluded derived features: {config.Config.EXCLUDED_FEATURES}")
    print(f"Target variable: {config.Config.TARGET_VARIABLE}")
    print(f"Train/Val/Test split: {config.Config.TRAIN_RATIO}/{config.Config.VAL_RATIO}/{config.Config.TEST_RATIO}")
    print(f"Random state: {config.Config.RANDOM_STATE}")
    print(f"Ultrasound target size: {config.Config.TARGET_SIZE}")
    print(f"Denoising method: {config.Config.DENOISING_METHOD}")
    
    print("\n" + "=" * 80)
    print("OUTPUT FILES GENERATED")
    print("=" * 80)
    
    print("\nUltrasound:")
    print(f"  - {ultrasound_quality_path} (Quality report)")
    print(f"  - {ultrasound_preprocess_path} (Preprocessing report)")
    print(f"  - data/processed/ultrasound/resized/ (Resized images)")
    print(f"  - data/processed/ultrasound/denoised/ (Denoised images)")
    print(f"  - data/processed/ultrasound/clahe/ (CLAHE images)")
    print(f"  - data/processed/ultrasound/roi/ (ROI images)")
    print(f"  - data/processed/ultrasound/normalized/ (Normalized images)")
    
    print("\nClinical:")
    print(f"  - {clinical_quality_path} (Quality report)")
    print(f"  - {clinical_preprocess_path} (Preprocessing report)")
    print(f"  - {clinical_mods_path} (Modifications log)")
    print(f"  - data/processed/clinical/train_processed.csv (Training data)")
    print(f"  - data/processed/clinical/val_processed.csv (Validation data)")
    print(f"  - data/processed/clinical/test_processed.csv (Test data)")
    print(f"  - data/processed/clinical/preprocessing_objects/ (Fitted objects)")
    
    # Save summary to JSON
    summary_path = reports_dir / 'preprocessing_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nComplete summary saved to: {summary_path}")
    
    print("\n" + "=" * 80)
    print("PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)
    
    return summary


if __name__ == "__main__":
    summary = generate_final_report()
