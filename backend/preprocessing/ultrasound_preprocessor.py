"""
Ultrasound Preprocessor
Implements the ultrasound preprocessing pipeline using PreprocessingConfig.

Pipeline:
    Original Image
        ↓
    Grayscale Conversion
        ↓
    Resize to 224 × 224
        ↓
    SRAD Speckle Noise Reduction
        ↓
    CLAHE Contrast Enhancement
        ↓
    Min-Max Normalization
        ↓
    Final Preprocessed Image
"""

import os
import logging
from typing import Optional, Tuple

import cv2
import numpy as np

from configs.preprocessing_config import PreprocessingConfig

class UltrasoundPreprocessor:
    """Preprocessing pipeline for ultrasound images."""

    def __init__(self, config=PreprocessingConfig):
        """
        Initialize the ultrasound preprocessor.

        Args:
            config: PreprocessingConfig class containing pipeline parameters.
        """
        self.config = config

        # Create output directories
        self.config.ensure_output_directories()

        # Configure logging
        self._setup_logging()

    # ============================================================
    # LOGGING
    # ============================================================

    def _setup_logging(self):
        """Configure logging for the preprocessing pipeline."""

        self.logger = logging.getLogger("UltrasoundPreprocessor")

        if not self.logger.handlers:
            self.logger.setLevel(
                getattr(logging, self.config.LOG_LEVEL)
            )

            formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s"
            )

            if self.config.LOG_TO_CONSOLE:
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(formatter)
                self.logger.addHandler(console_handler)

            if self.config.LOG_FILE:
                file_handler = logging.FileHandler(
                    self.config.LOG_FILE
                )
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)

    # ============================================================
    # IMAGE LOADING
    # ============================================================

    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load an ultrasound image.

        Args:
            image_path: Path to input image.

        Returns:
            Image as NumPy array.
        """

        if not os.path.exists(image_path):
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        image = cv2.imread(
            image_path,
            cv2.IMREAD_UNCHANGED
        )

        if image is None:
            raise ValueError(
                f"Unable to read image: {image_path}"
            )

        self.logger.info(
            f"Loaded image: {image_path}"
        )

        return image

    # ============================================================
    # GRAYSCALE CONVERSION
    # ============================================================

    def convert_to_grayscale(
        self,
        image: np.ndarray
    ) -> np.ndarray:
        """
        Convert image to grayscale if enabled in configuration.

        Args:
            image: Input image.

        Returns:
            Grayscale image.
        """

        if not self.config.CONVERT_TO_GRAYSCALE:
            return image

        # Already grayscale
        if len(image.shape) == 2:
            return image

        # BGRA → grayscale
        if image.shape[2] == 4:
            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGRA2GRAY
            )

        # BGR → grayscale
        elif image.shape[2] == 3:
            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY
            )

        else:
            raise ValueError(
                f"Unsupported number of channels: "
                f"{image.shape[2]}"
            )

        self.logger.debug(
            "Converted image to grayscale"
        )

        return image

    # ============================================================
    # RESIZING
    # ============================================================

    def resize_image(
        self,
        image: np.ndarray
    ) -> np.ndarray:
        """
        Resize image according to configuration.

        Supports aspect-ratio-preserving resize with padding
        or direct resizing to the target dimensions.
        """

        target_width = self.config.TARGET_WIDTH
        target_height = self.config.TARGET_HEIGHT

        if not self.config.MAINTAIN_ASPECT_RATIO:

            resized = cv2.resize(
                image,
                (target_width, target_height),
                interpolation=self._get_interpolation()
            )

            return resized

        # Original dimensions
        height, width = image.shape[:2]

        if width == 0 or height == 0:
            raise ValueError(
                "Invalid image dimensions."
            )

        # Scale while maintaining aspect ratio
        scale = min(
            target_width / width,
            target_height / height
        )

        new_width = max(
            1,
            int(round(width * scale))
        )

        new_height = max(
            1,
            int(round(height * scale))
        )

        resized = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=self._get_interpolation()
        )

        # Create padded output
        padded = np.full(
            (target_height, target_width),
            self.config.PADDING_COLOR,
            dtype=resized.dtype
        )

        # Center image
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2

        padded[
            y_offset:y_offset + new_height,
            x_offset:x_offset + new_width
        ] = resized

        return padded

    def _get_interpolation(self) -> int:
        """Convert interpolation configuration to OpenCV constant."""

        interpolation = self.config.RESIZE_INTERPOLATION.lower()

        mapping = {
            "nearest": cv2.INTER_NEAREST,
            "linear": cv2.INTER_LINEAR,
            "cubic": cv2.INTER_CUBIC,
            "lanczos": cv2.INTER_LANCZOS4
        }

        if interpolation not in mapping:
            raise ValueError(
                f"Unsupported interpolation method: "
                f"{interpolation}"
            )

        return mapping[interpolation]

    # ============================================================
    # SRAD SPECKLE NOISE REDUCTION
    # ============================================================

    def apply_srad(
        self,
        image: np.ndarray
    ) -> np.ndarray:
        """
        Apply Speckle Reducing Anisotropic Diffusion (SRAD).

        Uses the configuration parameters:
            SRAD_ITERATIONS
            SRAD_TIME_STEP
            SRAD_Q0
            SRAD_KERNEL_SIZE
            SRAD_Q0_ESTIMATION_REGION
        """

        # Convert to float
        img = image.astype(np.float32)

        # Normalize temporarily to [0, 1]
        min_val = np.min(img)
        max_val = np.max(img)

        if max_val > min_val:
            img = (img - min_val) / (
                max_val - min_val
            )
        else:
            return image.copy()

        iterations = self.config.SRAD_ITERATIONS
        dt = self.config.SRAD_TIME_STEP
        kernel_size = self.config.SRAD_KERNEL_SIZE

        # Estimate q0 automatically if required
        q0 = self.config.SRAD_Q0

        if q0 is None:
            q0 = self._estimate_q0(
                img
            )

        q0 = max(float(q0), 1e-6)

        for _ in range(iterations):

            # Local statistics
            mean = cv2.blur(
                img,
                (kernel_size, kernel_size)
            )

            mean_sq = cv2.blur(
                img * img,
                (kernel_size, kernel_size)
            )

            variance = np.maximum(
                mean_sq - mean * mean,
                0
            )

            coefficient_of_variation_sq = (
                variance /
                (mean * mean + 1e-8)
            )

            # SRAD diffusion coefficient
            numerator = (
                coefficient_of_variation_sq
                - q0 * q0
            )

            denominator = (
                q0 * q0 *
                (1.0 + q0 * q0)
                + 1e-8
            )

            diffusion = 1.0 / (
                1.0 +
                np.maximum(
                    numerator / denominator,
                    0
                )
            )

            # Approximate diffusion using Laplacian
            laplacian = cv2.Laplacian(
                img,
                cv2.CV_32F
            )

            img = img + (
                dt * diffusion * laplacian
            )

            img = np.clip(
                img,
                0.0,
                1.0
            )

        # Convert back to 8-bit
        result = (
            img * 255.0
        ).astype(np.uint8)

        return result

    def _estimate_q0(
        self,
        image: np.ndarray
    ) -> float:
        """
        Estimate SRAD speckle scale parameter q0.
        """

        region = (
            self.config.SRAD_Q0_ESTIMATION_REGION
            .lower()
        )

        height, width = image.shape

        if region == "center":

            h_start = height // 4
            h_end = 3 * height // 4

            w_start = width // 4
            w_end = 3 * width // 4

            sample = image[
                h_start:h_end,
                w_start:w_end
            ]

        elif region == "corners":

            size = min(
                height,
                width
            ) // 4

            samples = [
                image[:size, :size],
                image[:size, -size:],
                image[-size:, :size],
                image[-size:, -size:]
            ]

            sample = np.concatenate(
                [s.flatten() for s in samples]
            )

        else:
            sample = image

        mean = np.mean(sample)
        std = np.std(sample)

        if mean < 1e-8:
            return 0.1

        q0 = std / mean

        return float(
            max(q0, 0.01)
        )

    # ============================================================
    # CLAHE
    # ============================================================

    def apply_clahe(
        self,
        image: np.ndarray
    ) -> np.ndarray:
        """
        Apply Contrast Limited Adaptive Histogram Equalization.
        """

        clahe = cv2.createCLAHE(
            clipLimit=self.config.CLAHE_CLIP_LIMIT,
            tileGridSize=self.config.CLAHE_TILE_SIZE
        )

        enhanced = clahe.apply(
            image
        )

        return enhanced

    # ============================================================
    # NORMALIZATION
    # ============================================================

    def normalize_image(
        self,
        image: np.ndarray
    ) -> np.ndarray:
        """
        Normalize image according to configuration.

        Current default:
            Min-Max normalization → [0, 1]
        """

        method = (
            self.config.NORMALIZATION_METHOD
            .lower()
        )

        image_float = image.astype(
            np.float32
        )

        if method == "minmax":

            old_min = np.min(
                image_float
            )

            old_max = np.max(
                image_float
            )

            if old_max == old_min:
                return np.full_like(
                    image_float,
                    self.config.NORMALIZATION_MIN,
                    dtype=np.float32
                )

            normalized = (
                (image_float - old_min)
                /
                (old_max - old_min)
            )

            new_min = (
                self.config.NORMALIZATION_MIN
            )

            new_max = (
                self.config.NORMALIZATION_MAX
            )

            normalized = (
                normalized *
                (new_max - new_min)
                + new_min
            )

            return normalized.astype(
                np.float32
            )

        elif method == "zscore":

            mean = np.mean(
                image_float
            )

            std = np.std(
                image_float
            )

            if std < 1e-8:
                return np.zeros_like(
                    image_float,
                    dtype=np.float32
                )

            return (
                (image_float - mean) / std
            ).astype(np.float32)

        elif method == "percentile":

            lower = np.percentile(
                image_float,
                self.config.PERCENTILE_LOWER
            )

            upper = np.percentile(
                image_float,
                self.config.PERCENTILE_UPPER
            )

            if upper <= lower:
                return np.zeros_like(
                    image_float,
                    dtype=np.float32
                )

            normalized = (
                (image_float - lower)
                /
                (upper - lower)
            )

            normalized = np.clip(
                normalized,
                0.0,
                1.0
            )

            return normalized.astype(
                np.float32
            )

        else:
            raise ValueError(
                f"Unsupported normalization method: "
                f"{method}"
            )

    # ============================================================
    # SAVE IMAGE
    # ============================================================

    def save_stage(
        self,
        image: np.ndarray,
        stage: str,
        image_id: str
    ) -> str:
        """
        Save an intermediate/final preprocessing stage.
        """

        output_path = (
            self.config.get_output_path(
                stage,
                image_id
            )
        )

        os.makedirs(
            os.path.dirname(output_path),
            exist_ok=True
        )

        # Normalized images are float [0,1].
        # PNG expects an integer representation,
        # so convert only when saving.
        if (
            image.dtype == np.float32
            or image.dtype == np.float64
        ):
            save_image = np.clip(
                image,
                0.0,
                1.0
            )

            save_image = (
                save_image * 255.0
            ).astype(np.uint8)

        else:
            save_image = image

        success = cv2.imwrite(
            output_path,
            save_image
        )

        if not success:
            raise IOError(
                f"Failed to save image: "
                f"{output_path}"
            )

        return output_path

    # ============================================================
    # COMPLETE PIPELINE
    # ============================================================

    def preprocess_image(
        self,
        image_path: str,
        image_id: Optional[str] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Run the complete ultrasound preprocessing pipeline.

        Pipeline:
            Load
            → Grayscale
            → Resize
            → SRAD
            → CLAHE
            → Min-Max normalization

        Returns:
            final_image:
                Normalized image as float32 in [0,1].

            stage_paths:
                Dictionary containing paths to saved stages.
        """

        if image_id is None:
            image_id = os.path.splitext(
                os.path.basename(image_path)
            )[0]

        stage_paths = {}

        try:

            # ----------------------------------------------------
            # 1. LOAD
            # ----------------------------------------------------

            image = self.load_image(
                image_path
            )

            stage_paths["original"] = (
                self.save_stage(
                    image,
                    "original",
                    image_id
                )
            )

            # ----------------------------------------------------
            # 2. GRAYSCALE
            # ----------------------------------------------------

            image = self.convert_to_grayscale(
                image
            )

            # ----------------------------------------------------
            # 3. RESIZE
            # ----------------------------------------------------

            image = self.resize_image(
                image
            )

            stage_paths["resized"] = (
                self.save_stage(
                    image,
                    "resized",
                    image_id
                )
            )

            # ----------------------------------------------------
            # 4. SRAD
            # ----------------------------------------------------

            if (
                self.config.NOISE_REDUCTION_METHOD
                .lower()
                == "srad"
            ):

                image = self.apply_srad(
                    image
                )

            elif (
                self.config.NOISE_REDUCTION_METHOD
                .lower()
                == "lee"
            ):

                image = self.apply_lee_filter(
                    image
                )

            else:

                raise ValueError(
                    "Unsupported noise reduction method: "
                    f"{self.config.NOISE_REDUCTION_METHOD}"
                )

            stage_paths["denoised"] = (
                self.save_stage(
                    image,
                    "denoised",
                    image_id
                )
            )

            # ----------------------------------------------------
            # 5. CLAHE
            # ----------------------------------------------------

            image = self.apply_clahe(
                image
            )

            stage_paths["clahe"] = (
                self.save_stage(
                    image,
                    "clahe",
                    image_id
                )
            )

            # ----------------------------------------------------
            # 6. NORMALIZATION
            # ----------------------------------------------------

            final_image = self.normalize_image(
                image
            )

            stage_paths["normalized"] = (
                self.save_stage(
                    final_image,
                    "normalized",
                    image_id
                )
            )

            # ----------------------------------------------------
            # LOGGING
            # ----------------------------------------------------

            self.logger.info(
                f"Preprocessing completed: {image_id}"
            )

            self.logger.info(
                f"Final shape: {final_image.shape}"
            )

            self.logger.info(
                f"Final range: "
                f"{final_image.min():.4f} - "
                f"{final_image.max():.4f}"
            )

            return final_image, stage_paths

        except Exception as e:

            self.logger.error(
                f"Preprocessing failed for "
                f"{image_id}: {str(e)}"
            )

            if not self.config.CONTINUE_ON_ERROR:
                raise

            return None, stage_paths

    # ============================================================
    # LEE FILTER
    # ============================================================

    def apply_lee_filter(
        self,
        image: np.ndarray
    ) -> np.ndarray:
        """
        Apply a Lee filter for speckle noise reduction.
        """

        img = image.astype(
            np.float32
        )

        kernel_size = (
            self.config.LEE_FILTER_SIZE
        )

        iterations = (
            self.config.LEE_FILTER_ITERATIONS
        )

        for _ in range(iterations):

            mean = cv2.blur(
                img,
                (kernel_size, kernel_size)
            )

            mean_sq = cv2.blur(
                img * img,
                (kernel_size, kernel_size)
            )

            variance = np.maximum(
                mean_sq - mean * mean,
                0
            )

            noise_variance = np.mean(
                variance
            )

            weight = (
                variance /
                (variance + noise_variance + 1e-8)
            )

            img = (
                mean +
                weight * (img - mean)
            )

        return np.clip(
            img,
            0,
            255
        ).astype(np.uint8)