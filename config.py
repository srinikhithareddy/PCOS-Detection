"""
Configuration file for PCOS preprocessing pipeline
"""

class Config:
    # Paths
    DATA_ROOT = "data"
    RAW_DATA = "data/raw"
    PROCESSED_DATA = "data/processed"
    REPORTS_DIR = "data/reports"
    
    # Ultrasound paths
    ULTRASOUND_RAW = "data/raw/UltraSound/Ovarian_US"
    ULTRASOUND_PROCESSED = "data/processed/ultrasound"
    
    # Clinical paths
    CLINICAL_RAW = "data/raw/Clinical/PCOS Dataset.xlsx"
    CLINICAL_PROCESSED = "data/processed/clinical"
    
    # Ultrasound classes
    ULTRASOUND_CLASSES = ["dominant_follicle", "healthy", "poly_cyst", "simple_cyst", "complex_cyst"]
    
    # Quality assessment thresholds (V2 - approved after visual validation)
    BRISQUE_GOOD_THRESHOLD = 15  # Lower is better
    BRISQUE_POOR_THRESHOLD = 35
    NIQE_GOOD_THRESHOLD = 5.5  # Lower is better
    NIQE_POOR_THRESHOLD = 8.0
    
    # Image preprocessing
    TARGET_SIZE = (512, 512)
    CLAHE_CLIP_LIMIT = 2.0
    CLAHE_TILE_SIZE = (8, 8)
    
    # Denoising method: 'lee' or 'srad'
    DENOISING_METHOD = 'lee'
    LEE_FILTER_SIZE = 3
    LEE_NUM_ITERATIONS = 1
    
    # Clinical preprocessing
    EXCLUDED_FEATURES = ['BMI', 'Waist_Hip_Ratio', 'LH_FSH_Ratio', 'HOMA_IR']
    TARGET_VARIABLE = 'PCOS_Diagnosis'
    
    # Data split
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    RANDOM_STATE = 42
    
    # Clinical features that should not have negative values
    NON_NEGATIVE_FEATURES = [
        'Height_cm', 'Weight_kg', 'BMI', 'Waist_Circumference_cm', 
        'Hip_Circumference_cm', 'Waist_Hip_Ratio', 'Age_at_Menarche',
        'Menstrual_Cycle_Length_days', 'Gravidity', 'Parity',
        'Hirsutism_Score_FG', 'Acne_Severity', 'Alopecia',
        'Skin_Darkening_Acanthosis', 'Blood_Pressure_Systolic',
        'Blood_Pressure_Diastolic', 'Physical_Activity_Level',
        'Smoking_Status', 'Alcohol_Intake', 'Dietary_Sugar_Intake',
        'Sleep_Hours', 'FSH_mIU_mL', 'LH_mIU_mL', 'LH_FSH_Ratio',
        'Total_Testosterone_ng_dL', 'Free_Testosterone_pg_mL',
        'DHEAS_ug_dL', 'Prolactin_ng_mL', 'Estradiol_pg_mL',
        'Progesterone_ng_mL', 'SHBG_nmol_L', 'Fasting_Glucose_mg_dL',
        'Fasting_Insulin_uIU_mL', 'HOMA_IR', 'HbA1c_percent',
        'Total_Cholesterol_mg_dL', 'HDL_mg_dL', 'LDL_mg_dL',
        'Triglycerides_mg_dL', 'Ovary_Volume_Left_cm3',
        'Ovary_Volume_Right_cm3', 'Follicle_Count_Left',
        'Follicle_Count_Right', 'CRP_mg_L', 'ALT_U_L', 'AST_U_L',
        'TSH_uIU_mL', 'Vitamin_D_ng_mL', 'Hemoglobin_g_dL'
    ]
    
    # Binary/categorical features (not continuous measurements)
    CATEGORICAL_FEATURES = [
        'Menstrual_Irregularity', 'Alopecia', 'Skin_Darkening_Acanthosis',
        'Smoking_Status', 'Alcohol_Intake'
    ]
