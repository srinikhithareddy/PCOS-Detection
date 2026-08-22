"""
Unit tests for ultrasound preprocessing module
"""

import os
import tempfile
import shutil

import pytest
import numpy as np
import cv2

from preprocessing.ultrasound_preprocessor import UltrasoundPreprocessor
from configs.preprocessing_config import PreprocessingConfig


class TestUltrasoundPreprocessor:
    """Test suite for UltrasoundPreprocessor"""

    # ============================================================
    # FIXTURE
    # ============================================================

    @pytest.fixture
    def sample_image_path(self):
        """Create a temporary sample ultrasound image."""

        temp_dir = tempfile.mkdtemp()

        image_path = os.path.join(
            temp_dir,
            "test_ultrasound.png"
        )

        # Create a sample grayscale image
        image = np.random.randint(
            0,
            255,
            (256, 256),
            dtype=np.uint8
        )

        cv2.imwrite(
            image_path,
            image
        )

        yield image_path

        shutil.rmtree(
            temp_dir,
            ignore_errors=True
        )

    @pytest.fixture
    def preprocessor(self):
        """Create UltrasoundPreprocessor with temporary output directory."""

        original_dir = PreprocessingConfig.OUTPUT_BASE_DIR

        temp_dir = tempfile.mkdtemp()

        PreprocessingConfig.OUTPUT_BASE_DIR = temp_dir

        preprocessor = UltrasoundPreprocessor()

        yield preprocessor

        shutil.rmtree(
            temp_dir,
            ignore_errors=True
        )

        PreprocessingConfig.OUTPUT_BASE_DIR = original_dir

    # ============================================================
    # INITIALIZATION
    # ============================================================

    def test_preprocessor_initialization(
        self,
        preprocessor
    ):
        """Test that preprocessor initializes correctly."""

        assert preprocessor is not None
        assert preprocessor.config is not None

    # ============================================================
    # IMAGE LOADING
    # ============================================================

    def test_load_image(
        self,
        preprocessor,
        sample_image_path
    ):
        """Test image loading."""

        image = preprocessor.load_image(
            sample_image_path
        )

        assert image is not None
        assert isinstance(
            image,
            np.ndarray
        )

        assert image.shape == (
            256,
            256
        )

    # ============================================================
    # GRAYSCALE
    # ============================================================

    def test_convert_to_grayscale(
        self,
        preprocessor
    ):
        """Test grayscale conversion."""

        test_image = np.random.randint(
            0,
            255,
            (256, 256, 3),
            dtype=np.uint8
        )

        grayscale = preprocessor.convert_to_grayscale(
            test_image
        )

        assert grayscale is not None
        assert isinstance(
            grayscale,
            np.ndarray
        )

        assert len(
            grayscale.shape
        ) == 2

    # ============================================================
    # RESIZING
    # ============================================================

    def test_resize_image(
        self,
        preprocessor
    ):
        """Test image resizing."""

        test_image = np.random.randint(
            0,
            255,
            (256, 256),
            dtype=np.uint8
        )

        resized = preprocessor.resize_image(
            test_image
        )

        assert resized.shape[0] == (
            PreprocessingConfig.TARGET_HEIGHT
        )

        assert resized.shape[1] == (
            PreprocessingConfig.TARGET_WIDTH
        )

    # ============================================================
    # ASPECT RATIO
    # ============================================================

    def test_resize_with_aspect_ratio(
        self,
        preprocessor
    ):
        """Test resizing with aspect ratio preservation."""

        original_value = (
            PreprocessingConfig.MAINTAIN_ASPECT_RATIO
        )

        PreprocessingConfig.MAINTAIN_ASPECT_RATIO = True

        try:

            test_image = np.random.randint(
                0,
                255,
                (100, 200),
                dtype=np.uint8
            )

            resized = preprocessor.resize_image(
                test_image
            )

            assert resized.shape == (
                PreprocessingConfig.TARGET_HEIGHT,
                PreprocessingConfig.TARGET_WIDTH
            )

        finally:

            PreprocessingConfig.MAINTAIN_ASPECT_RATIO = (
                original_value
            )

    # ============================================================
    # SRAD
    # ============================================================

    def test_srad_filter(
        self,
        preprocessor
    ):
        """Test SRAD speckle noise reduction."""

        test_image = np.random.randint(
            0,
            255,
            (256, 256),
            dtype=np.uint8
        )

        denoised = preprocessor.apply_srad(
            test_image
        )

        assert denoised is not None

        assert isinstance(
            denoised,
            np.ndarray
        )

        assert denoised.shape == (
            256,
            256
        )

        assert denoised.dtype == np.uint8

    # ============================================================
    # LEE FILTER
    # ============================================================

    def test_lee_filter(
        self,
        preprocessor
    ):
        """Test Lee speckle noise reduction."""

        test_image = np.random.randint(
            0,
            255,
            (256, 256),
            dtype=np.uint8
        )

        denoised = preprocessor.apply_lee_filter(
            test_image
        )

        assert denoised is not None

        assert isinstance(
            denoised,
            np.ndarray
        )

        assert denoised.shape == (
            256,
            256
        )

        assert denoised.dtype == np.uint8

    # ============================================================
    # CLAHE
    # ============================================================

    def test_clahe(
        self,
        preprocessor
    ):
        """Test CLAHE contrast enhancement."""

        test_image = np.random.randint(
            0,
            255,
            (256, 256),
            dtype=np.uint8
        )

        enhanced = preprocessor.apply_clahe(
            test_image
        )

        assert enhanced is not None

        assert isinstance(
            enhanced,
            np.ndarray
        )

        assert enhanced.shape == (
            256,
            256
        )

        assert enhanced.dtype == np.uint8

    # ============================================================
    # MIN-MAX NORMALIZATION
    # ============================================================

    def test_normalization_minmax(
        self,
        preprocessor
    ):
        """Test Min-Max normalization."""

        original_method = (
            PreprocessingConfig.NORMALIZATION_METHOD
        )

        PreprocessingConfig.NORMALIZATION_METHOD = (
            "minmax"
        )

        try:

            test_image = np.random.randint(
                0,
                255,
                (256, 256),
                dtype=np.uint8
            )

            normalized = preprocessor.normalize_image(
                test_image
            )

            assert normalized is not None

            assert normalized.shape == (
                256,
                256
            )

            assert normalized.dtype == np.float32

            assert normalized.min() >= 0.0
            assert normalized.max() <= 1.0

        finally:

            PreprocessingConfig.NORMALIZATION_METHOD = (
                original_method
            )

    # ============================================================
    # Z-SCORE NORMALIZATION
    # ============================================================

    def test_normalization_zscore(
        self,
        preprocessor
    ):
        """Test Z-score normalization."""

        original_method = (
            PreprocessingConfig.NORMALIZATION_METHOD
        )

        PreprocessingConfig.NORMALIZATION_METHOD = (
            "zscore"
        )

        try:

            test_image = np.random.randint(
                0,
                255,
                (256, 256),
                dtype=np.uint8
            )

            normalized = preprocessor.normalize_image(
                test_image
            )

            assert normalized is not None

            assert normalized.shape == (
                256,
                256
            )

            assert normalized.dtype == np.float32

        finally:

            PreprocessingConfig.NORMALIZATION_METHOD = (
                original_method
            )

    # ============================================================
    # PERCENTILE NORMALIZATION
    # ============================================================

    def test_normalization_percentile(
        self,
        preprocessor
    ):
        """Test percentile normalization."""

        original_method = (
            PreprocessingConfig.NORMALIZATION_METHOD
        )

        PreprocessingConfig.NORMALIZATION_METHOD = (
            "percentile"
        )

        try:

            test_image = np.random.randint(
                0,
                255,
                (256, 256),
                dtype=np.uint8
            )

            normalized = preprocessor.normalize_image(
                test_image
            )

            assert normalized is not None

            assert normalized.shape == (
                256,
                256
            )

            assert normalized.dtype == np.float32

            assert normalized.min() >= 0.0
            assert normalized.max() <= 1.0

        finally:

            PreprocessingConfig.NORMALIZATION_METHOD = (
                original_method
            )

    # ============================================================
    # SAVE STAGE
    # ============================================================

    def test_save_stage(
        self,
        preprocessor
    ):
        """Test saving an intermediate preprocessing stage."""

        test_image = np.random.randint(
            0,
            255,
            (256, 256),
            dtype=np.uint8
        )

        output_path = preprocessor.save_stage(
            test_image,
            "resized",
            "test_001"
        )

        assert output_path is not None

        assert os.path.exists(
            output_path
        )

    # ============================================================
    # COMPLETE PIPELINE
    # ============================================================

    def test_complete_preprocessing_pipeline(
        self,
        preprocessor,
        sample_image_path
    ):
        """Test complete ultrasound preprocessing pipeline."""

        final_image, stage_paths = (
            preprocessor.preprocess_image(
                sample_image_path,
                image_id="test_001"
            )
        )

        assert final_image is not None

        assert isinstance(
            final_image,
            np.ndarray
        )

        assert final_image.dtype == np.float32

        assert final_image.shape == (
            PreprocessingConfig.TARGET_HEIGHT,
            PreprocessingConfig.TARGET_WIDTH
        )

        assert final_image.min() >= 0.0
        assert final_image.max() <= 1.0

        assert isinstance(
            stage_paths,
            dict
        )

        # Check all expected stages
        assert "original" in stage_paths
        assert "resized" in stage_paths
        assert "denoised" in stage_paths
        assert "clahe" in stage_paths
        assert "normalized" in stage_paths

    # ============================================================
    # STAGE OUTPUT SAVING
    # ============================================================

    def test_stage_output_saving(
        self,
        preprocessor,
        sample_image_path
    ):
        """Test that all preprocessing stages are saved."""

        final_image, stage_paths = (
            preprocessor.preprocess_image(
                sample_image_path,
                image_id="test_002"
            )
        )

        assert final_image is not None

        for stage, path in stage_paths.items():

            assert os.path.exists(
                path
            ), f"Output file for {stage} does not exist: {path}"


# ================================================================
# PREPROCESSING CONFIG TESTS
# ================================================================

class TestPreprocessingConfig:
    """Test suite for PreprocessingConfig."""

    def test_config_initialization(self):
        """Test that config initializes correctly."""

        config = PreprocessingConfig()

        assert config is not None

    def test_get_resize_params(self):
        """Test resize parameters."""

        params = (
            PreprocessingConfig.get_resize_params()
        )

        assert "target_width" in params
        assert "target_height" in params
        assert "interpolation" in params

        assert params["target_width"] == (
            PreprocessingConfig.TARGET_WIDTH
        )

        assert params["target_height"] == (
            PreprocessingConfig.TARGET_HEIGHT
        )

    def test_get_noise_reduction_params(self):
        """Test noise reduction parameters."""

        params = (
            PreprocessingConfig
            .get_noise_reduction_params()
        )

        assert "method" in params

        assert params["method"] in [
            "lee",
            "srad"
        ]

    def test_get_clahe_params(self):
        """Test CLAHE parameters."""

        params = (
            PreprocessingConfig
            .get_clahe_params()
        )

        assert "clip_limit" in params
        assert "tile_size" in params

    def test_get_normalization_params(self):
        """Test normalization parameters."""

        params = (
            PreprocessingConfig
            .get_normalization_params()
        )

        assert "method" in params

        assert params["method"] in [
            "minmax",
            "zscore",
            "percentile"
        ]

    def test_get_output_path(self):
        """Test output path generation."""

        image_id = "test_image"
        stage = "resized"

        path = (
            PreprocessingConfig.get_output_path(
                stage,
                image_id
            )
        )

        assert image_id in path
        assert stage in path

    def test_ensure_output_directories(self):
        """Test creation of output directories."""

        original_dir = (
            PreprocessingConfig.OUTPUT_BASE_DIR
        )

        temp_dir = tempfile.mkdtemp()

        PreprocessingConfig.OUTPUT_BASE_DIR = (
            temp_dir
        )

        try:

            PreprocessingConfig.ensure_output_directories()

            for stage_dir in (
                PreprocessingConfig.STAGE_DIRECTORIES.values()
            ):

                dir_path = os.path.join(
                    temp_dir,
                    stage_dir
                )

                assert os.path.exists(
                    dir_path
                )

        finally:

            shutil.rmtree(
                temp_dir,
                ignore_errors=True
            )

            PreprocessingConfig.OUTPUT_BASE_DIR = (
                original_dir
            )


# ================================================================
# RUN TESTS
# ================================================================

if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v"
    ])