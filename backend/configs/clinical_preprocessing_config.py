"""
Clinical Preprocessing Configuration
Configurable parameters for clinical data preprocessing pipeline
"""

from typing import Dict, Any, List
import os


class ClinicalPreprocessingConfig:
    """Configuration for clinical data preprocessing"""
    
    # ===== MISSING VALUE HANDLING CONFIGURATION =====
    
    # Missing value imputation strategy: 'mean', 'median', 'mode', 'knn', 'constant'
    NUMERICAL_IMPUTATION_STRATEGY = 'median'
    
    # Constant value for constant imputation
    NUMERICAL_IMPUTATION_CONSTANT = 0
    
    # Categorical imputation strategy: 'most_frequent', 'constant', 'missing'
    CATEGORICAL_IMPUTATION_STRATEGY = 'most_frequent'
    
    # Constant value for categorical imputation
    CATEGORICAL_IMPUTATION_CONSTANT = 'missing'
    
    # KNN imputation parameters
    KNN_NEIGHBORS = 5
    
    # Features that should NOT be imputed (must be present)
    CRITICAL_FEATURES = [
        'Age',
        'BMI',
        'LH_FSH_Ratio',
        'Total_Testosterone_ng_dL'
    ]
    
    # ===== OUTLIER HANDLING CONFIGURATION =====
    
    # Outlier handling strategy: 'clip', 'remove', 'winsorize', 'none'
    OUTLIER_HANDLING_STRATEGY = 'clip'
    
    # Outlier detection method: 'iqr', 'zscore', 'isolation_forest'
    OUTLIER_DETECTION_METHOD = 'iqr'
    
    # IQR multiplier for outlier detection
    IQR_MULTIPLIER = 1.5
    
    # Z-score threshold for outlier detection
    Z_SCORE_THRESHOLD = 3.0
    
    # Winsorize percentiles (lower, upper)
    WINSORIZE_PERCENTILES = (5, 95)
    
    # Features to skip outlier handling (preserve clinical extremes)
    OUTLIER_EXEMPT_FEATURES = [
        'Age',  # Age extremes may be clinically relevant
        'Total_Testosterone_ng_dL',  # High testosterone may indicate PCOS
        'LH_FSH_Ratio'  # High ratio is diagnostic for PCOS
    ]
    
    # ===== RANGE & CONSISTENCY VALIDATION CONFIGURATION =====
    
    # Whether to enforce range constraints
    ENFORCE_RANGE_CONSTRAINTS = True
    
    # Whether to enforce consistency checks
    ENFORCE_CONSISTENCY_CHECKS = True
    
    # Action on range violation: 'clip', 'flag', 'none'
    RANGE_VIOLATION_ACTION = 'clip'
    
    # Action on consistency violation: 'clip', 'flag', 'none'
    CONSISTENCY_VIOLATION_ACTION = 'clip'
    
    # Tolerance for calculated vs reported values (percentage)
    CALCULATION_TOLERANCE_PERCENT = 10.0
    
    # ===== CATEGORICAL ENCODING CONFIGURATION =====
    
    # Encoding method: 'onehot', 'label', 'target'
    CATEGORICAL_ENCODING_METHOD = 'onehot'
    
    # Handle unknown categories: 'ignore', 'error', 'use_default'
    UNKNOWN_CATEGORY_HANDLING = 'ignore'
    
    # Drop first column in one-hot encoding (to avoid multicollinearity)
    DROP_FIRST_ONEHOT = True
    
    # Categorical features (to be identified automatically or specified)
    CATEGORICAL_FEATURES = [
        'Menstrual_Irregularity',
        'Hirsutism_Score_FG',
        'Acne_Severity',
        'Alopecia',
        'Skin_Darkening_Acanthosis',
        'Physical_Activity_Level',
        'Smoking_Status',
        'Alcohol_Intake',
        'Dietary_Sugar_Intake'
    ]
    
    # ===== FEATURE SELECTION CONFIGURATION =====
    
    # Feature selection method: 'variance_threshold', 'correlation', 'mutual_info', 'recursive', 'none'
    FEATURE_SELECTION_METHOD = 'variance_threshold'
    
    # Variance threshold for variance-based selection
    VARIANCE_THRESHOLD = 0.01
    
    # Correlation threshold for correlation-based selection
    CORRELATION_THRESHOLD = 0.95
    
    # Number of features to select for recursive selection
    N_FEATURES_TO_SELECT = 30
    
    # Mutual information threshold
    MUTUAL_INFO_THRESHOLD = 0.01
    
    # Features to always include (force selection)
    FORCE_INCLUDE_FEATURES = [
        'Age',
        'BMI',
        'LH_FSH_Ratio',
        'Total_Testosterone_ng_dL'
    ]
    
    # ===== NORMALIZATION CONFIGURATION =====
    
    # Normalization method: 'standard', 'minmax', 'robust', 'none'
    NORMALIZATION_METHOD = 'standard'
    
    # Min-max normalization range
    MINMAX_RANGE = (0, 1)
    
    # Robust scaler quantile range
    ROBUST_QUANTILE_RANGE = (25.0, 75.0)
    
    # Features to skip normalization
    SKIP_NORMALIZATION_FEATURES = []
    
    # ===== DATA LEAKAGE PREVENTION CONFIGURATION =====
    
    # Directory to save fitted preprocessing objects
    PREPROCESSING_OBJECTS_DIR = 'preprocessing_objects'
    
    # File prefix for saved objects
    OBJECTS_FILE_PREFIX = 'clinical_preprocessing'
    
    # ===== LOGGING CONFIGURATION =====
    
    # Log level: 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    LOG_LEVEL = 'INFO'
    
    # Log file path
    LOG_FILE = 'clinical_preprocessing.log'
    
    # Log to console
    LOG_TO_CONSOLE = True
    
    # ===== CLINICAL FEATURE DEFINITIONS =====
    
    # All 47 clinical features
    ALL_CLINICAL_FEATURES = [
        # Demographics
        'Age',
        'Height_cm',
        'Weight_kg',
        'BMI',
        'Waist_Circumference_cm',
        'Hip_Circumference_cm',
        'Waist_Hip_Ratio',
        
        # Menstrual & Reproductive
        'Age_at_Menarche',
        'Menstrual_Cycle_Length_days',
        'Menstrual_Irregularity',
        'Gravidity',
        'Parity',
        
        # Physical Signs
        'Hirsutism_Score_FG',
        'Acne_Severity',
        'Alopecia',
        'Skin_Darkening_Acanthosis',
        
        # Vitals & Lifestyle
        'Blood_Pressure_Systolic',
        'Blood_Pressure_Diastolic',
        'Physical_Activity_Level',
        'Smoking_Status',
        'Alcohol_Intake',
        'Dietary_Sugar_Intake',
        'Sleep_Hours',
        
        # Hormonal Panel
        'FSH_mIU_mL',
        'LH_mIU_mL',
        'LH_FSH_Ratio',
        'Total_Testosterone_ng_dL',
        'Free_Testosterone_pg_mL',
        'DHEAS_ug_dL',
        'Prolactin_ng_mL',
        'Estradiol_pg_mL',
        'Progesterone_ng_mL',
        'SHBG_nmol_L',
        
        # Metabolic Panel
        'Fasting_Glucose_mg_dL',
        'Fasting_Insulin_uIU_mL',
        'HOMA_IR',
        'HbA1c_percent',
        'Total_Cholesterol_mg_dL',
        'HDL_mg_dL',
        'LDL_mg_dL',
        'Triglycerides_mg_dL',
        
        # Other Labs
        'CRP_mg_L',
        'ALT_U_L',
        'AST_U_L',
        'TSH_uIU_mL',
        'Vitamin_D_ng_mL',
        'Hemoglobin_g_dL',
    ]
    
    @classmethod
    def get_imputation_params(cls) -> Dict[str, Any]:
        """Get imputation parameters"""
        return {
            'numerical_strategy': cls.NUMERICAL_IMPUTATION_STRATEGY,
            'numerical_constant': cls.NUMERICAL_IMPUTATION_CONSTANT,
            'categorical_strategy': cls.CATEGORICAL_IMPUTATION_STRATEGY,
            'categorical_constant': cls.CATEGORICAL_IMPUTATION_CONSTANT,
            'knn_neighbors': cls.KNN_NEIGHBORS,
            'critical_features': cls.CRITICAL_FEATURES
        }
    
    @classmethod
    def get_outlier_params(cls) -> Dict[str, Any]:
        """Get outlier handling parameters"""
        return {
            'handling_strategy': cls.OUTLIER_HANDLING_STRATEGY,
            'detection_method': cls.OUTLIER_DETECTION_METHOD,
            'iqr_multiplier': cls.IQR_MULTIPLIER,
            'z_score_threshold': cls.Z_SCORE_THRESHOLD,
            'winsorize_percentiles': cls.WINSORIZE_PERCENTILES,
            'exempt_features': cls.OUTLIER_EXEMPT_FEATURES
        }
    
    @classmethod
    def get_encoding_params(cls) -> Dict[str, Any]:
        """Get categorical encoding parameters"""
        return {
            'method': cls.CATEGORICAL_ENCODING_METHOD,
            'unknown_handling': cls.UNKNOWN_CATEGORY_HANDLING,
            'drop_first': cls.DROP_FIRST_ONEHOT,
            'categorical_features': cls.CATEGORICAL_FEATURES
        }
    
    @classmethod
    def get_feature_selection_params(cls) -> Dict[str, Any]:
        """Get feature selection parameters"""
        return {
            'method': cls.FEATURE_SELECTION_METHOD,
            'variance_threshold': cls.VARIANCE_THRESHOLD,
            'correlation_threshold': cls.CORRELATION_THRESHOLD,
            'n_features_to_select': cls.N_FEATURES_TO_SELECT,
            'mutual_info_threshold': cls.MUTUAL_INFO_THRESHOLD,
            'force_include': cls.FORCE_INCLUDE_FEATURES
        }
    
    @classmethod
    def get_normalization_params(cls) -> Dict[str, Any]:
        """Get normalization parameters"""
        return {
            'method': cls.NORMALIZATION_METHOD,
            'minmax_range': cls.MINMAX_RANGE,
            'robust_quantile_range': cls.ROBUST_QUANTILE_RANGE,
            'skip_features': cls.SKIP_NORMALIZATION_FEATURES
        }
    
    @classmethod
    def get_objects_file_path(cls, mode: str = 'train') -> str:
        """
        Get path for saving/loading preprocessing objects
        
        Args:
            mode: 'train' or 'inference'
            
        Returns:
            Path to preprocessing objects file
        """
        os.makedirs(cls.PREPROCESSING_OBJECTS_DIR, exist_ok=True)
        filename = f"{cls.OBJECTS_FILE_PREFIX}_{mode}.pkl"
        return os.path.join(cls.PREPROCESSING_OBJECTS_DIR, filename)
    
    @classmethod
    def ensure_directories(cls) -> None:
        """Create necessary directories"""
        os.makedirs(cls.PREPROCESSING_OBJECTS_DIR, exist_ok=True)
