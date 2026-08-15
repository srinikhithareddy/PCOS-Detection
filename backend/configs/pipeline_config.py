"""
End-to-End Preprocessing Pipeline Configuration
Configuration for the complete preprocessing pipeline integrating all stages
"""

from typing import Dict, Any, List
import os


class PipelineConfig:
    """Configuration for end-to-end preprocessing pipeline"""
    
    # ===== PIPELINE CONTROL =====
    
    # Whether to run ultrasound preprocessing
    RUN_ULTRASOUND_PIPELINE = True
    
    # Whether to run clinical preprocessing
    RUN_CLINICAL_PIPELINE = True
    
    # Whether to require both modalities (True) or allow single modality (False)
    REQUIRE_BOTH_MODALITIES = False
    
    # Minimum quality threshold to proceed with preprocessing
    MINIMUM_QUALITY_SCORE = 30.0  # Below this, preprocessing is skipped
    
    # ===== OUTPUT CONFIGURATION =====
    
    # Base directory for pipeline outputs
    OUTPUT_BASE_DIR = "pipeline_outputs"
    
    # Subdirectories for different outputs
    OUTPUT_SUBDIRS = {
        'raw_data': 'raw_data',
        'quality_reports': 'quality_reports',
        'ultrasound_preprocessed': 'ultrasound_preprocessed',
        'clinical_preprocessed': 'clinical_preprocessed',
        'final_outputs': 'final_outputs',
        'logs': 'logs'
    }
    
    # Output format for images
    IMAGE_OUTPUT_FORMAT = 'png'
    
    # Output format for clinical data
    CLINICAL_OUTPUT_FORMAT = 'parquet'  # 'parquet' or 'csv'
    
    # ===== LOGGING CONFIGURATION =====
    
    # Log level: 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    LOG_LEVEL = 'INFO'
    
    # Log file path
    LOG_FILE = 'pipeline.log'
    
    # Log to console
    LOG_TO_CONSOLE = True
    
    # Log format
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
    
    # ===== VALIDATION CONFIGURATION =====
    
    # Whether to validate all outputs
    VALIDATE_OUTPUTS = True
    
    # Validation checks
    VALIDATION_CHECKS = {
        'ultrasound_roi': True,
        'segmentation_mask': True,
        'clinical_vector': True,
        'quality_reports': True
    }
    
    # Minimum ROI size (pixels)
    MIN_ROI_SIZE = 100
    
    # Maximum ROI size (pixels)
    MAX_ROI_SIZE = 500000
    
    # Minimum clinical vector length
    MIN_CLINICAL_VECTOR_LENGTH = 1
    
    # Maximum clinical vector length
    MAX_CLINICAL_VECTOR_LENGTH = 1000
    
    # ===== REPRODUCIBILITY CONFIGURATION =====
    
    # Random seed for reproducibility
    RANDOM_SEED = 42
    
    # Whether to save preprocessing metadata
    SAVE_METADATA = True
    
    # Metadata format: 'json' or 'yaml'
    METADATA_FORMAT = 'json'
    
    # ===== ERROR HANDLING CONFIGURATION =====
    
    # Continue pipeline if one modality fails
    CONTINUE_ON_MODALITY_FAILURE = True
    
    # Stop pipeline on critical errors
    STOP_ON_CRITICAL_ERRORS = True
    
    # Critical error types
    CRITICAL_ERRORS = [
        'memory_error',
        'file_not_found',
        'permission_error'
    ]
    
    # ===== PERFORMANCE CONFIGURATION =====
    
    # Maximum memory usage (GB)
    MAX_MEMORY_GB = 8
    
    # Number of parallel workers for preprocessing
    N_WORKERS = 1
    
    # Whether to use GPU for segmentation (if available)
    USE_GPU_IF_AVAILABLE = True
    
    @classmethod
    def get_output_path(cls, output_type: str, patient_id: str, filename: str = None) -> str:
        """
        Get output path for a specific type of output
        
        Args:
            output_type: Type of output (raw_data, quality_reports, etc.)
            patient_id: Patient/sample identifier
            filename: Optional filename (auto-generated if None)
            
        Returns:
            Full path to output file
        """
        subdir = cls.OUTPUT_SUBDIRS.get(output_type, output_type)
        
        if filename is None:
            filename = f"{patient_id}.{cls.IMAGE_OUTPUT_FORMAT if output_type in ['ultrasound_preprocessed'] else cls.CLINICAL_OUTPUT_FORMAT}"
        
        return os.path.join(cls.OUTPUT_BASE_DIR, subdir, filename)
    
    @classmethod
    def ensure_output_directories(cls) -> None:
        """Create all output directories if they don't exist"""
        for subdir in cls.OUTPUT_SUBDIRS.values():
            dir_path = os.path.join(cls.OUTPUT_BASE_DIR, subdir)
            os.makedirs(dir_path, exist_ok=True)
    
    @classmethod
    def get_pipeline_config(cls) -> Dict[str, Any]:
        """
        Get complete pipeline configuration
        
        Returns:
            Dictionary of all pipeline configuration parameters
        """
        return {
            'run_ultrasound': cls.RUN_ULTRASOUND_PIPELINE,
            'run_clinical': cls.RUN_CLINICAL_PIPELINE,
            'require_both_modalities': cls.REQUIRE_BOTH_MODALITIES,
            'minimum_quality_score': cls.MINIMUM_QUALITY_SCORE,
            'output_base_dir': cls.OUTPUT_BASE_DIR,
            'validate_outputs': cls.VALIDATE_OUTPUTS,
            'random_seed': cls.RANDOM_SEED,
            'continue_on_modality_failure': cls.CONTINUE_ON_MODALITY_FAILURE
        }
