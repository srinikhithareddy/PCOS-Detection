"""
Quality Assessment Configuration
Configurable thresholds for ultrasound and clinical data quality assessment
"""

from typing import Dict, Any


class QualityConfig:
    """Configuration for data quality assessment thresholds"""
    
    # ===== ULTRASOUND IMAGE QUALITY CONFIGURATION =====
    
    # BRISQUE score thresholds (lower is better, typical range 0-100)
    BRISQUE_GOOD_THRESHOLD = 25.0
    BRISQUE_POOR_THRESHOLD = 50.0
    BRISQUE_UNUSABLE_THRESHOLD = 75.0
    
    # NIQE score thresholds (lower is better, typical range 0-100)
    NIQE_GOOD_THRESHOLD = 20.0
    NIQE_POOR_THRESHOLD = 40.0
    NIQE_UNUSABLE_THRESHOLD = 60.0
    
    # Which metric to use as primary: 'brisque' or 'niqe' or 'both'
    PRIMARY_IMAGE_QUALITY_METRIC = 'brisque'
    
    # Minimum image dimensions for usability
    MIN_IMAGE_WIDTH = 64
    MIN_IMAGE_HEIGHT = 64
    MAX_IMAGE_WIDTH = 4096
    MAX_IMAGE_HEIGHT = 4096
    
    # Maximum file size in bytes (10 MB default)
    MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024
    
    # Allowed image formats
    ALLOWED_IMAGE_FORMATS = ['jpg', 'jpeg', 'png', 'bmp']
    
    # ===== CLINICAL DATA QUALITY CONFIGURATION =====
    
    # Missing value thresholds
    MAX_MISSING_VALUES_PERCENT = 20.0  # Maximum % of missing values allowed
    CRITICAL_FEATURES_MUST_BE_PRESENT = [
        'Age',
        'BMI',
        'LH_FSH_Ratio',
        'Total_Testosterone_ng_dL'
    ]
    
    # Outlier detection method: 'iqr', 'zscore', or 'isolation_forest'
    OUTLIER_DETECTION_METHOD = 'iqr'
    
    # IQR multiplier for outlier detection
    IQR_MULTIPLIER = 1.5
    
    # Z-score threshold for outlier detection
    Z_SCORE_THRESHOLD = 3.0
    
    # Clinical feature ranges (where medically established)
    # These are configurable and should be adjusted based on clinical guidelines
    CLINICAL_FEATURE_RANGES: Dict[str, Dict[str, float]] = {
        # Demographics
        'Age': {'min': 10.0, 'max': 60.0},
        'Height_cm': {'min': 100.0, 'max': 220.0},
        'Weight_kg': {'min': 30.0, 'max': 200.0},
        'BMI': {'min': 10.0, 'max': 60.0},
        'Waist_Circumference_cm': {'min': 40.0, 'max': 180.0},
        'Hip_Circumference_cm': {'min': 50.0, 'max': 200.0},
        'Waist_Hip_Ratio': {'min': 0.4, 'max': 1.2},
        
        # Menstrual & Reproductive
        'Age_at_Menarche': {'min': 8.0, 'max': 18.0},
        'Menstrual_Cycle_Length_days': {'min': 15.0, 'max': 45.0},
        'Gravidity': {'min': 0.0, 'max': 15.0},
        'Parity': {'min': 0.0, 'max': 10.0},
        
        # Physical Signs
        'Hirsutism_Score_FG': {'min': 0.0, 'max': 36.0},
        
        # Vitals
        'Blood_Pressure_Systolic': {'min': 70.0, 'max': 250.0},
        'Blood_Pressure_Diastolic': {'min': 40.0, 'max': 150.0},
        'Sleep_Hours': {'min': 3.0, 'max': 12.0},
        
        # Hormonal Panel
        'FSH_mIU_mL': {'min': 0.1, 'max': 50.0},
        'LH_mIU_mL': {'min': 0.1, 'max': 100.0},
        'LH_FSH_Ratio': {'min': 0.1, 'max': 10.0},
        'Total_Testosterone_ng_dL': {'min': 5.0, 'max': 200.0},
        'Free_Testosterone_pg_mL': {'min': 0.1, 'max': 50.0},
        'DHEAS_ug_dL': {'min': 10.0, 'max': 700.0},
        'Prolactin_ng_mL': {'min': 1.0, 'max': 200.0},
        'Estradiol_pg_mL': {'min': 10.0, 'max': 1000.0},
        'Progesterone_ng_mL': {'min': 0.1, 'max': 30.0},
        'SHBG_nmol_L': {'min': 5.0, 'max': 200.0},
        
        # Metabolic Panel
        'Fasting_Glucose_mg_dL': {'min': 50.0, 'max': 400.0},
        'Fasting_Insulin_uIU_mL': {'min': 2.0, 'max': 100.0},
        'HOMA_IR': {'min': 0.1, 'max': 10.0},
        'HbA1c_percent': {'min': 3.0, 'max': 15.0},
        'Total_Cholesterol_mg_dL': {'min': 100.0, 'max': 500.0},
        'HDL_mg_dL': {'min': 20.0, 'max': 150.0},
        'LDL_mg_dL': {'min': 30.0, 'max': 300.0},
        'Triglycerides_mg_dL': {'min': 30.0, 'max': 1000.0},
        
        # Other Labs
        'CRP_mg_L': {'min': 0.1, 'max': 50.0},
        'ALT_U_L': {'min': 5.0, 'max': 500.0},
        'AST_U_L': {'min': 5.0, 'max': 500.0},
        'TSH_uIU_mL': {'min': 0.1, 'max': 20.0},
        'Vitamin_D_ng_mL': {'min': 5.0, 'max': 150.0},
        'Hemoglobin_g_dL': {'min': 8.0, 'max': 20.0},
    }
    
    # Consistency check rules
    CONSISTENCY_RULES = {
        'BMI_calculation': True,  # Verify BMI matches height/weight
        'LH_FSH_calculation': True,  # Verify LH/FSH ratio matches individual values
        'Waist_Hip_calculation': True,  # Verify waist-hip ratio matches measurements
        'HOMA_IR_calculation': True,  # Verify HOMA-IR matches glucose/insulin
    }
    
    # Tolerance for calculated vs reported values (percentage)
    CALCULATION_TOLERANCE_PERCENT = 10.0
    
    # Reliability score weights
    RELIABILITY_WEIGHTS = {
        'missing_values': 0.3,
        'outliers': 0.25,
        'range_violations': 0.25,
        'consistency_violations': 0.2,
    }
    
    @classmethod
    def get_image_quality_thresholds(cls) -> Dict[str, Dict[str, float]]:
        """Get image quality thresholds"""
        return {
            'brisque': {
                'good': cls.BRISQUE_GOOD_THRESHOLD,
                'poor': cls.BRISQUE_POOR_THRESHOLD,
                'unusable': cls.BRISQUE_UNUSABLE_THRESHOLD
            },
            'niqe': {
                'good': cls.NIQE_GOOD_THRESHOLD,
                'poor': cls.NIQE_POOR_THRESHOLD,
                'unusable': cls.NIQE_UNUSABLE_THRESHOLD
            }
        }
    
    @classmethod
    def get_feature_range(cls, feature_name: str) -> Dict[str, float]:
        """Get valid range for a clinical feature"""
        return cls.CLINICAL_FEATURE_RANGES.get(feature_name, {})
    
    @classmethod
    def update_threshold(cls, category: str, metric: str, value: float) -> None:
        """Update a specific threshold value"""
        if category == 'brisque':
            if metric == 'good':
                cls.BRISQUE_GOOD_THRESHOLD = value
            elif metric == 'poor':
                cls.BRISQUE_POOR_THRESHOLD = value
            elif metric == 'unusable':
                cls.BRISQUE_UNUSABLE_THRESHOLD = value
        elif category == 'niqe':
            if metric == 'good':
                cls.NIQE_GOOD_THRESHOLD = value
            elif metric == 'poor':
                cls.NIQE_POOR_THRESHOLD = value
            elif metric == 'unusable':
                cls.NIQE_UNUSABLE_THRESHOLD = value
