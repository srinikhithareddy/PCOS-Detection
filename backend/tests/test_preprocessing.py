"""
Unit tests for ultrasound preprocessing module
"""

import pytest
import numpy as np
from PIL import Image
import io
import os
import tempfile
import shutil

from preprocessing.ultrasound_preprocessor import UltrasoundPreprocessor, PreprocessingStage
from configs.preprocessing_config import PreprocessingConfig


class TestUltrasoundPreprocessor:
    """Test suite for UltrasoundPreprocessor"""
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample test image"""
        img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        img = Image.fromarray(img_array, 'RGB')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        return img_bytes.read()
    
    @pytest.fixture
    def preprocessor(self):
        """Create an instance of UltrasoundPreprocessor with temp directory"""
        # Use a temporary directory for outputs
        original_dir = PreprocessingConfig.OUTPUT_BASE_DIR
        temp_dir = tempfile.mkdtemp()
        PreprocessingConfig.OUTPUT_BASE_DIR = temp_dir
        
        preprocessor = UltrasoundPreprocessor()
        
        yield preprocessor
        
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
        PreprocessingConfig.OUTPUT_BASE_DIR = original_dir
    
    def test_preprocessor_initialization(self, preprocessor):
        """Test that preprocessor initializes correctly"""
        assert preprocessor is not None
        assert preprocessor.config is not None
    
    def test_load_image(self, preprocessor, sample_image):
        """Test image loading"""
        image = preprocessor._load_image(sample_image)
        
        assert image is not None
        assert isinstance(image, np.ndarray)
        assert image.shape[2] == 3  # BGR format
    
    def test_resize_image(self, preprocessor):
        """Test image resizing"""
        # Create test image
        test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        resized = preprocessor._resize_image(test_image)
        
        assert resized.shape[0] == PreprocessingConfig.TARGET_HEIGHT
        assert resized.shape[1] == PreprocessingConfig.TARGET_WIDTH
    
    def test_resize_with_aspect_ratio(self, preprocessor):
        """Test resizing with aspect ratio preservation"""
        # Temporarily enable aspect ratio preservation
        original_value = PreprocessingConfig.MAINTAIN_ASPECT_RATIO
        PreprocessingConfig.MAINTAIN_ASPECT_RATIO = True
        
        test_image = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        resized = preprocessor._resize_image(test_image)
        
        assert resized.shape[0] == PreprocessingConfig.TARGET_HEIGHT
        assert resized.shape[1] == PreprocessingConfig.TARGET_WIDTH
        
        # Restore original value
        PreprocessingConfig.MAINTAIN_ASPECT_RATIO = original_value
    
    def test_lee_filter(self, preprocessor):
        """Test Lee filter for speckle noise reduction"""
        test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        params = {'filter_size': 5, 'iterations': 1}
        denoised = preprocessor._apply_lee_filter(test_image, params)
        
        assert denoised is not None
        assert denoised.shape == test_image.shape
        assert denoised.dtype == np.uint8
    
    def test_srad_filter(self, preprocessor):
        """Test SRAD filter (should fall back to Lee if fails)"""
        test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        params = {
            'iterations': 10,
            'time_step': 0.05,
            'conductance': 0.1
        }
        
        # SRAD may fail and fall back to Lee, which is acceptable
        denoised = preprocessor._apply_srad(test_image, params)
        
        assert denoised is not None
        assert denoised.shape == test_image.shape
    
    def test_clahe(self, preprocessor):
        """Test CLAHE application"""
        test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        enhanced = preprocessor._apply_clahe(test_image)
        
        assert enhanced is not None
        assert enhanced.shape == test_image.shape
        assert enhanced.dtype == np.uint8
    
    def test_segmentation(self, preprocessor):
        """Test ovarian follicle segmentation"""
        test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        mask = preprocessor._segment_ovarian_follicles(test_image)
        
        assert mask is not None
        assert mask.dtype == np.uint8
        # Mask should be binary or close to binary
        assert len(mask.shape) == 2  # Should be grayscale
    
    def test_roi_extraction(self, preprocessor):
        """Test ROI extraction from segmentation mask"""
        test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        # Create a simple mask with a region
        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[100:200, 100:200] = 255
        
        roi = preprocessor._extract_roi(test_image, mask)
        
        assert roi is not None
        assert roi.shape[0] <= test_image.shape[0]
        assert roi.shape[1] <= test_image.shape[1]
    
    def test_roi_extraction_no_contours(self, preprocessor):
        """Test ROI extraction when no contours found"""
        test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        # Empty mask
        mask = np.zeros((512, 512), dtype=np.uint8)
        
        roi = preprocessor._extract_roi(test_image, mask)
        
        # Should return original image if no contours
        assert roi.shape == test_image.shape
    
    def test_normalization_minmax(self, preprocessor):
        """Test min-max normalization"""
        test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        # Temporarily set method to minmax
        original_method = PreprocessingConfig.NORMALIZATION_METHOD
        PreprocessingConfig.NORMALIZATION_METHOD = 'minmax'
        
        normalized = preprocessor._normalize_image(test_image)
        
        assert normalized is not None
        assert normalized.shape == test_image.shape
        assert normalized.dtype == np.uint8
        
        # Restore original method
        PreprocessingConfig.NORMALIZATION_METHOD = original_method
    
    def test_normalization_zscore(self, preprocessor):
        """Test z-score normalization"""
        test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        # Temporarily set method to zscore
        original_method = PreprocessingConfig.NORMALIZATION_METHOD
        PreprocessingConfig.NORMALIZATION_METHOD = 'zscore'
        
        normalized = preprocessor._normalize_image(test_image)
        
        assert normalized is not None
        assert normalized.shape == test_image.shape
        
        # Restore original method
        PreprocessingConfig.NORMALIZATION_METHOD = original_method
    
    def test_normalization_percentile(self, preprocessor):
        """Test percentile normalization"""
        test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        # Temporarily set method to percentile
        original_method = PreprocessingConfig.NORMALIZATION_METHOD
        PreprocessingConfig.NORMALIZATION_METHOD = 'percentile'
        
        normalized = preprocessor._normalize_image(test_image)
        
        assert normalized is not None
        assert normalized.shape == test_image.shape
        
        # Restore original method
        PreprocessingConfig.NORMALIZATION_METHOD = original_method
    
    def test_complete_preprocessing_pipeline(self, preprocessor, sample_image):
        """Test complete preprocessing pipeline"""
        result = preprocessor.preprocess_image(
            image_data=sample_image,
            image_id="test_001",
            filename="test_image.png"
        )
        
        assert result is not None
        assert result['image_id'] == "test_001"
        assert result['filename'] == "test_image.png"
        assert 'stages_completed' in result
        assert 'stage_outputs' in result
        assert 'preprocessing_status' in result
    
    def test_stage_output_saving(self, preprocessor, sample_image):
        """Test that stage outputs are saved correctly"""
        image_id = "test_002"
        
        result = preprocessor.preprocess_image(
            image_data=sample_image,
            image_id=image_id,
            filename="test_image.png"
        )
        
        # Check that output files exist
        for stage in result['stage_outputs'].values():
            assert os.path.exists(stage), f"Output file {stage} does not exist"


class TestPreprocessingConfig:
    """Test suite for PreprocessingConfig"""
    
    def test_config_initialization(self):
        """Test that config initializes correctly"""
        config = PreprocessingConfig()
        assert config is not None
    
    def test_get_resize_params(self):
        """Test getting resize parameters"""
        params = PreprocessingConfig.get_resize_params()
        
        assert 'target_width' in params
        assert 'target_height' in params
        assert 'interpolation' in params
        assert params['target_width'] == PreprocessingConfig.TARGET_WIDTH
        assert params['target_height'] == PreprocessingConfig.TARGET_HEIGHT
    
    def test_get_noise_reduction_params(self):
        """Test getting noise reduction parameters"""
        params = PreprocessingConfig.get_noise_reduction_params()
        
        assert 'method' in params
        assert params['method'] in ['lee', 'srad']
    
    def test_get_clahe_params(self):
        """Test getting CLAHE parameters"""
        params = PreprocessingConfig.get_clahe_params()
        
        assert 'clip_limit' in params
        assert 'tile_size' in params
    
    def test_get_unet_params(self):
        """Test getting U-Net parameters"""
        params = PreprocessingConfig.get_unet_params()
        
        assert 'input_size' in params
        assert 'num_classes' in params
        assert 'filters' in params
    
    def test_get_normalization_params(self):
        """Test getting normalization parameters"""
        params = PreprocessingConfig.get_normalization_params()
        
        assert 'method' in params
        assert params['method'] in ['minmax', 'zscore', 'percentile']
    
    def test_get_output_path(self):
        """Test getting output path for a stage"""
        image_id = "test_image"
        stage = "resized"
        
        path = PreprocessingConfig.get_output_path(stage, image_id)
        
        assert image_id in path
        assert stage in path
    
    def test_ensure_output_directories(self):
        """Test creation of output directories"""
        # Use temp directory
        original_dir = PreprocessingConfig.OUTPUT_BASE_DIR
        temp_dir = tempfile.mkdtemp()
        PreprocessingConfig.OUTPUT_BASE_DIR = temp_dir
        
        PreprocessingConfig.ensure_output_directories()
        
        # Check that directories were created
        for stage_dir in PreprocessingConfig.STAGE_DIRECTORIES.values():
            dir_path = os.path.join(temp_dir, stage_dir)
            assert os.path.exists(dir_path)
        
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
        PreprocessingConfig.OUTPUT_BASE_DIR = original_dir


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
