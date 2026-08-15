"""
Unit tests for segmentation module
"""

import pytest
import numpy as np
import tempfile
import shutil
import os

from segmentation import SegmentationInference
from models.unet import UNet, create_unet_model
from preprocessing_config import PreprocessingConfig


class TestUNetArchitecture:
    """Test suite for U-Net architecture"""
    
    def test_unet_initialization(self):
        """Test U-Net initialization"""
        unet = UNet()
        assert unet is not None
        assert unet.input_size == (512, 512)
        assert unet.num_classes == 1
        assert unet.filters == 64
        assert unet.depth == 4
    
    def test_unet_custom_initialization(self):
        """Test U-Net with custom parameters"""
        unet = UNet(
            input_size=(256, 256),
            num_classes=2,
            filters=32,
            depth=3,
            dropout_rate=0.2
        )
        assert unet.input_size == (256, 256)
        assert unet.num_classes == 2
        assert unet.filters == 32
        assert unet.depth == 3
        assert unet.dropout_rate == 0.2
    
    def test_build_model(self):
        """Test building U-Net model"""
        unet = UNet()
        model = unet.build_model()
        
        assert model is not None
        assert unet.model is not None
        assert len(model.input_shape) == 4  # (batch, height, width, channels)
    
    def test_model_output_shape(self):
        """Test model output shape"""
        unet = UNet(input_size=(256, 256), num_classes=1)
        model = unet.build_model()
        
        # Input shape should be (None, 256, 256, 3)
        assert model.input_shape == (None, 256, 256, 3)
        
        # Output shape should be (None, 256, 256, 1)
        assert model.output_shape == (None, 256, 256, 1)
    
    def test_multiclass_output_shape(self):
        """Test multi-class output shape"""
        unet = UNet(input_size=(256, 256), num_classes=3)
        model = unet.build_model()
        
        assert model.output_shape == (None, 256, 256, 3)
    
    def test_compile_model(self):
        """Test model compilation"""
        unet = UNet()
        model = unet.compile_model()
        
        assert model is not None
        assert model.optimizer is not None
        assert model.loss is not None
    
    def test_convenience_function(self):
        """Test convenience function for creating U-Net"""
        model = create_unet_model()
        
        assert model is not None
        assert model.input_shape == (None, 512, 512, 3)
        assert model.output_shape == (None, 512, 512, 1)


class TestSegmentationInference:
    """Test suite for SegmentationInference"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample test image"""
        return np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    @pytest.fixture
    def sample_mask(self):
        """Create a sample segmentation mask"""
        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[100:200, 100:200] = 255  # Simple square region
        return mask
    
    def test_inference_initialization_no_weights(self, temp_dir):
        """Test initialization without weights"""
        original_dir = PreprocessingConfig.OUTPUT_BASE_DIR
        PreprocessingConfig.OUTPUT_BASE_DIR = temp_dir
        
        inference = SegmentationInference(weights_path=None)
        
        assert inference is not None
        assert inference.model_loaded is False
        assert inference.model is None
        
        PreprocessingConfig.OUTPUT_BASE_DIR = original_dir
    
    def test_inference_initialization_invalid_weights(self, temp_dir):
        """Test initialization with invalid weights path"""
        original_dir = PreprocessingConfig.OUTPUT_BASE_DIR
        PreprocessingConfig.OUTPUT_BASE_DIR = temp_dir
        
        inference = SegmentationInference(weights_path="/nonexistent/path.h5")
        
        assert inference is not None
        assert inference.model_loaded is False
        
        PreprocessingConfig.OUTPUT_BASE_DIR = original_dir
    
    def test_segmentation_without_model(self, temp_dir, sample_image):
        """Test segmentation when model is not loaded"""
        original_dir = PreprocessingConfig.OUTPUT_BASE_DIR
        PreprocessingConfig.OUTPUT_BASE_DIR = temp_dir
        
        inference = SegmentationInference(weights_path=None)
        mask, metadata = inference.segment_image(sample_image)
        
        assert mask is None
        assert metadata['status'] == 'failed'
        assert 'Model not loaded' in metadata['error']
        
        PreprocessingConfig.OUTPUT_BASE_DIR = original_dir
    
    def test_preprocess_for_model(self, temp_dir, sample_image):
        """Test image preprocessing for model"""
        original_dir = PreprocessingConfig.OUTPUT_BASE_DIR
        PreprocessingConfig.OUTPUT_BASE_DIR = temp_dir
        
        inference = SegmentationInference(weights_path=None)
        preprocessed = inference._preprocess_for_model(sample_image)
        
        assert preprocessed.shape == (1, 512, 512, 3)
        assert preprocessed.dtype == np.float32
        assert np.max(preprocessed) <= 1.0
        assert np.min(preprocessed) >= 0.0
        
        PreprocessingConfig.OUTPUT_BASE_DIR = original_dir
    
    def test_postprocess_prediction(self, temp_dir):
        """Test prediction postprocessing"""
        original_dir = PreprocessingConfig.OUTPUT_BASE_DIR
        PreprocessingConfig.OUTPUT_BASE_DIR = temp_dir
        
        inference = SegmentationInference(weights_path=None)
        
        # Create mock prediction
        prediction = np.random.rand(1, 512, 512, 1).astype(np.float32)
        mask = inference._postprocess_prediction(prediction)
        
        assert mask is not None
        assert mask.shape == (512, 512)
        assert mask.dtype == np.uint8
        assert np.unique(mask).tolist() in [[0, 255], [0], [255]]  # Binary
        
        PreprocessingConfig.OUTPUT_BASE_DIR = original_dir
    
    def test_extract_roi_with_contours(self, temp_dir, sample_image, sample_mask):
        """Test ROI extraction with valid contours"""
        original_dir = PreprocessingConfig.OUTPUT_BASE_DIR
        PreprocessingConfig.OUTPUT_BASE_DIR = temp_dir
        
        inference = SegmentationInference(weights_path=None)
        roi, metadata = inference.extract_roi(sample_image, sample_mask)
        
        assert roi is not None
        assert metadata['status'] == 'success'
        assert metadata['bbox'] is not None
        assert metadata['roi_used_full_image'] is False
        assert metadata['area'] > 0
        
        PreprocessingConfig.OUTPUT_BASE_DIR = original_dir
    
    def test_extract_roi_no_contours(self, temp_dir, sample_image):
        """Test ROI extraction with no contours"""
        original_dir = PreprocessingConfig.OUTPUT_BASE_DIR
        PreprocessingConfig.OUTPUT_BASE_DIR = temp_dir
        
        inference = SegmentationInference(weights_path=None)
        empty_mask = np.zeros((512, 512), dtype=np.uint8)
        
        roi, metadata = inference.extract_roi(sample_image, empty_mask)
        
        assert roi is not None
        assert metadata['status'] == 'no_contours'
        assert metadata['bbox'] is None
        assert metadata['roi_used_full_image'] is True
        
        PreprocessingConfig.OUTPUT_BASE_DIR = original_dir
    
    def test_extract_roi_too_small(self, temp_dir, sample_image):
        """Test ROI extraction with too small region"""
        original_dir = PreprocessingConfig.OUTPUT_BASE_DIR
        PreprocessingConfig.OUTPUT_BASE_DIR = temp_dir
        
        inference = SegmentationInference(weights_path=None)
        
        # Create very small mask
        small_mask = np.zeros((512, 512), dtype=np.uint8)
        small_mask[100:101, 100:101] = 255  # 1x1 pixel
        
        roi, metadata = inference.extract_roi(sample_image, small_mask)
        
        assert roi is not None
        assert metadata['status'] == 'roi_too_small'
        assert metadata['roi_used_full_image'] is True
        
        PreprocessingConfig.OUTPUT_BASE_DIR = original_dir
    
    def test_create_overlay(self, temp_dir, sample_image, sample_mask):
        """Test overlay creation"""
        original_dir = PreprocessingConfig.OUTPUT_BASE_DIR
        PreprocessingConfig.OUTPUT_BASE_DIR = temp_dir
        
        inference = SegmentationInference(weights_path=None)
        overlay = inference.create_overlay(sample_image, sample_mask, alpha=0.5)
        
        assert overlay is not None
        assert overlay.shape == sample_image.shape
        assert overlay.dtype == np.uint8
        
        PreprocessingConfig.OUTPUT_BASE_DIR = original_dir
    
    def test_normalize_roi_minmax(self, temp_dir, sample_image):
        """Test ROI normalization with minmax"""
        original_dir = PreprocessingConfig.OUTPUT_BASE_DIR
        PreprocessingConfig.OUTPUT_BASE_DIR = temp_dir
        
        inference = SegmentationInference(weights_path=None)
        
        # Temporarily set method to minmax
        original_method = PreprocessingConfig.NORMALIZATION_METHOD
        PreprocessingConfig.NORMALIZATION_METHOD = 'minmax'
        
        normalized = inference.normalize_roi(sample_image)
        
        assert normalized is not None
        assert normalized.shape == sample_image.shape
        assert normalized.dtype == np.uint8
        
        # Restore original method
        PreprocessingConfig.NORMALIZATION_METHOD = original_method
        PreprocessingConfig.OUTPUT_BASE_DIR = original_dir
    
    def test_normalize_roi_zscore(self, temp_dir, sample_image):
        """Test ROI normalization with zscore"""
        original_dir = PreprocessingConfig.OUTPUT_BASE_DIR
        PreprocessingConfig.OUTPUT_BASE_DIR = temp_dir
        
        inference = SegmentationInference(weights_path=None)
        
        # Temporarily set method to zscore
        original_method = PreprocessingConfig.NORMALIZATION_METHOD
        PreprocessingConfig.NORMALIZATION_METHOD = 'zscore'
        
        normalized = inference.normalize_roi(sample_image)
        
        assert normalized is not None
        assert normalized.shape == sample_image.shape
        
        # Restore original method
        PreprocessingConfig.NORMALIZATION_METHOD = original_method
        PreprocessingConfig.OUTPUT_BASE_DIR = original_dir
    
    def test_save_segmentation_outputs(self, temp_dir, sample_image, sample_mask):
        """Test saving segmentation outputs"""
        original_dir = PreprocessingConfig.OUTPUT_BASE_DIR
        PreprocessingConfig.OUTPUT_BASE_DIR = temp_dir
        
        inference = SegmentationInference(weights_path=None)
        
        image_id = "test_image"
        roi = sample_image[100:200, 100:200]
        overlay = inference.create_overlay(sample_image, sample_mask)
        
        bbox_info = {'x': 100, 'y': 100, 'width': 100, 'height': 100}
        metadata = {'status': 'success'}
        
        output_paths = inference.save_segmentation_outputs(
            image_id=image_id,
            image=sample_image,
            mask=sample_mask,
            roi=roi,
            overlay=overlay,
            bbox_info=bbox_info,
            metadata=metadata
        )
        
        assert 'original' in output_paths
        assert 'mask' in output_paths
        assert 'roi' in output_paths
        assert 'overlay' in output_paths
        assert 'normalized' in output_paths
        assert 'bbox_info' in output_paths
        
        # Check that files exist
        for path in output_paths.values():
            if path.endswith('.json'):
                assert os.path.exists(path)
            else:
                assert os.path.exists(path)
        
        PreprocessingConfig.OUTPUT_BASE_DIR = original_dir


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
