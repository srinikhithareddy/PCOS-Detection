"""
Ultrasound Preprocessing Configuration
Configurable parameters for ultrasound image preprocessing pipeline
"""

from typing import Dict, Any
import os


class PreprocessingConfig:
    """Configuration for ultrasound image preprocessing"""
    
    # ===== RESIZING CONFIGURATION =====
    
    # Target resolution for preprocessing
    TARGET_WIDTH = 512
    TARGET_HEIGHT = 512
    
    # Interpolation method: 'linear', 'cubic', 'nearest', 'lanczos'
    RESIZE_INTERPOLATION = 'linear'
    
    # Maintain aspect ratio (True) or force exact dimensions (False)
    MAINTAIN_ASPECT_RATIO = True
    
    # Padding color when maintaining aspect ratio
    PADDING_COLOR = 0  # Black
    
    # ===== SPECKLE NOISE REDUCTION CONFIGURATION =====
    
    # Noise reduction method: 'lee' or 'srad'
    NOISE_REDUCTION_METHOD = 'srad'
    
    # Lee filter parameters
    LEE_FILTER_SIZE = 5  # Kernel size (must be odd)
    LEE_FILTER_ITERATIONS = 1  # Number of iterations
    
    # SRAD parameters (Speckle Reducing Anisotropic Diffusion)
    SRAD_ITERATIONS = 5  # Number of diffusion iterations (conservative initial test)
    SRAD_TIME_STEP = 0.05  # Time step for numerical stability (0.01-0.25)
    SRAD_Q0 = None  # Speckle scale parameter (None = auto-estimate from image)
    SRAD_KERNEL_SIZE = 3  # Kernel size for local statistics (3-5)
    SRAD_Q0_ESTIMATION_REGION = 'center'  # 'center', 'corners', or 'full' for q0 estimation
    
    # ===== CLAHE CONFIGURATION =====
    
    # CLAHE clip limit (contrast enhancement)
    CLAHE_CLIP_LIMIT = 2.0
    
    # CLAHE tile grid size (for local contrast)
    CLAHE_TILE_SIZE = (8, 8)
    
    # ===== U-NET SEGMENTATION CONFIGURATION =====
    
    # U-Net model parameters
    UNET_INPUT_SIZE = (512, 512)
    UNET_NUM_CLASSES = 1  # Binary segmentation (ovarian follicle vs background)
    UNET_FILTERS = 64  # Number of filters in first layer
    UNET_DEPTH = 4  # Depth of U-Net encoder
    UNET_DROPOUT_RATE = 0.1
    
    # Model weights path (if using pre-trained model)
    UNET_MODEL_PATH = None  # Path to pre-trained weights
    
    # Segmentation threshold
    SEGMENTATION_THRESHOLD = 0.5
    
    # Minimum ROI size (pixels)
    MIN_ROI_AREA = 1000
    
    # Maximum ROI size (pixels)
    MAX_ROI_AREA = 200000
    
    # ===== NORMALIZATION CONFIGURATION =====
    
    # Normalization method: 'minmax', 'zscore', 'percentile'
    NORMALIZATION_METHOD = 'minmax'
    
    # Min-max normalization range
    NORMALIZATION_MIN = 0.0
    NORMALIZATION_MAX = 1.0
    
    # Z-score parameters
    Z_SCORE_MEAN = 0.0
    Z_SCORE_STD = 1.0
    
    # Percentile normalization
    PERCENTILE_LOWER = 1.0
    PERCENTILE_UPPER = 99.0
    
    # ===== OUTPUT DIRECTORY CONFIGURATION =====
    
    # Base directory for preprocessing outputs
    OUTPUT_BASE_DIR = "preprocessing_outputs"
    
    # Subdirectories for each stage
    STAGE_DIRECTORIES = {
        'original': 'original',
        'resized': 'resized',
        'denoised': 'denoised',
        'clahe': 'clahe_enhanced',
        'segmentation': 'segmentation',
        'roi': 'roi_extracted',
        'normalized': 'normalized'
    }
    
    # Image format for outputs
    OUTPUT_FORMAT = 'png'
    
    # Quality for lossy formats (if using JPEG)
    OUTPUT_QUALITY = 95
    
    # ===== LOGGING CONFIGURATION =====
    
    # Log level: 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    LOG_LEVEL = 'INFO'
    
    # Log file path
    LOG_FILE = 'preprocessing.log'
    
    # Log to console
    LOG_TO_CONSOLE = True
    
    # ===== ERROR HANDLING CONFIGURATION =====
    
    # Continue processing on error (True) or stop (False)
    CONTINUE_ON_ERROR = False
    
    # Save intermediate results even if later stages fail
    SAVE_INTERMEDIATE_ON_ERROR = True
    
    @classmethod
    def get_output_path(cls, stage: str, image_id: str) -> str:
        """
        Get output path for a specific preprocessing stage
        
        Args:
            stage: Preprocessing stage name
            image_id: Image identifier
            
        Returns:
            Full path to output file
        """
        stage_dir = cls.STAGE_DIRECTORIES.get(stage, stage)
        filename = f"{image_id}.{cls.OUTPUT_FORMAT}"
        return os.path.join(cls.OUTPUT_BASE_DIR, stage_dir, filename)
    
    @classmethod
    def ensure_output_directories(cls) -> None:
        """Create all output directories if they don't exist"""
        for stage_dir in cls.STAGE_DIRECTORIES.values():
            dir_path = os.path.join(cls.OUTPUT_BASE_DIR, stage_dir)
            os.makedirs(dir_path, exist_ok=True)
    
    @classmethod
    def get_resize_params(cls) -> Dict[str, Any]:
        """Get resizing parameters"""
        return {
            'target_width': cls.TARGET_WIDTH,
            'target_height': cls.TARGET_HEIGHT,
            'interpolation': cls.RESIZE_INTERPOLATION,
            'maintain_aspect_ratio': cls.MAINTAIN_ASPECT_RATIO,
            'padding_color': cls.PADDING_COLOR
        }
    
    @classmethod
    def get_noise_reduction_params(cls) -> Dict[str, Any]:
        """Get noise reduction parameters"""
        if cls.NOISE_REDUCTION_METHOD == 'lee':
            return {
                'method': 'lee',
                'filter_size': cls.LEE_FILTER_SIZE,
                'iterations': cls.LEE_FILTER_ITERATIONS
            }
        elif cls.NOISE_REDUCTION_METHOD == 'srad':
            return {
                'method': 'srad',
                'iterations': cls.SRAD_ITERATIONS,
                'time_step': cls.SRAD_TIME_STEP,
                'q0': cls.SRAD_Q0,
                'kernel_size': cls.SRAD_KERNEL_SIZE,
                'q0_estimation_region': cls.SRAD_Q0_ESTIMATION_REGION
            }
        else:
            return {'method': cls.NOISE_REDUCTION_METHOD}
    
    @classmethod
    def get_clahe_params(cls) -> Dict[str, Any]:
        """Get CLAHE parameters"""
        return {
            'clip_limit': cls.CLAHE_CLIP_LIMIT,
            'tile_size': cls.CLAHE_TILE_SIZE
        }
    
    @classmethod
    def get_unet_params(cls) -> Dict[str, Any]:
        """Get U-Net parameters"""
        return {
            'input_size': cls.UNET_INPUT_SIZE,
            'num_classes': cls.UNET_NUM_CLASSES,
            'filters': cls.UNET_FILTERS,
            'depth': cls.UNET_DEPTH,
            'dropout_rate': cls.UNET_DROPOUT_RATE,
            'model_path': cls.UNET_MODEL_PATH,
            'segmentation_threshold': cls.SEGMENTATION_THRESHOLD
        }
    
    @classmethod
    def get_normalization_params(cls) -> Dict[str, Any]:
        """Get normalization parameters"""
        return {
            'method': cls.NORMALIZATION_METHOD,
            'min': cls.NORMALIZATION_MIN,
            'max': cls.NORMALIZATION_MAX,
            'z_score_mean': cls.Z_SCORE_MEAN,
            'z_score_std': cls.Z_SCORE_STD,
            'percentile_lower': cls.PERCENTILE_LOWER,
            'percentile_upper': cls.PERCENTILE_UPPER
        }
