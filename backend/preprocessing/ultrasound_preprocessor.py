"""
Ultrasound Image Preprocessing Pipeline
Implements the complete preprocessing pipeline for ultrasound images
"""

import os
import io
import logging
from typing import Dict, Any, Optional, Tuple
from enum import Enum
import numpy as np
from PIL import Image
import cv2

from configs.preprocessing_config import PreprocessingConfig

# Configure logging
logging.basicConfig(
    level=getattr(logging, PreprocessingConfig.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(PreprocessingConfig.LOG_FILE),
        logging.StreamHandler() if PreprocessingConfig.LOG_TO_CONSOLE else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)


class PreprocessingStage(Enum):
    """Preprocessing pipeline stages"""
    ORIGINAL = "original"
    RESIZED = "resized"
    DENOISED = "denoised"
    CLAHE = "clahe_enhanced"
    SEGMENTATION = "segmentation"
    ROI = "roi_extracted"
    NORMALIZED = "normalized"


class UltrasoundPreprocessor:
    """Ultrasound image preprocessing pipeline"""
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        Initialize the preprocessor
        
        Args:
            config: Preprocessing configuration (uses default if None)
        """
        self.config = config or PreprocessingConfig()
        self.config.ensure_output_directories()
        
        # Initialize U-Net model (will be loaded when needed)
        self.unet_model = None
        
        logger.info("UltrasoundPreprocessor initialized")
    
    def preprocess_image(
        self,
        image_data: bytes,
        image_id: str,
        filename: str
    ) -> Dict[str, Any]:
        """
        Run complete preprocessing pipeline on an ultrasound image
        
        Args:
            image_data: Raw image bytes
            image_id: Unique identifier for the image
            filename: Original filename
            
        Returns:
            Dictionary containing preprocessing results and stage outputs
        """
        try:
            logger.info(f"Starting preprocessing for image {image_id}")
            
            # Stage 0: Save original
            original_image = self._load_image(image_data)
            if original_image is None:
                raise ValueError("Failed to load image")
            
            self._save_stage_output(
                original_image, 
                image_id, 
                PreprocessingStage.ORIGINAL
            )
            
            results = {
                'image_id': image_id,
                'filename': filename,
                'stages_completed': [],
                'stage_outputs': {},
                'final_image': None,
                'preprocessing_status': 'in_progress'
            }
            
            # Stage 1: Resize
            try:
                resized_image = self._resize_image(original_image)
                self._save_stage_output(resized_image, image_id, PreprocessingStage.RESIZED)
                results['stage_outputs']['resized'] = self._get_stage_output_path(image_id, PreprocessingStage.RESIZED)
                results['stages_completed'].append('resize')
                logger.info(f"Resize completed for {image_id}")
            except Exception as e:
                logger.error(f"Resize failed for {image_id}: {str(e)}")
                if not self.config.CONTINUE_ON_ERROR:
                    raise
            
            # Stage 2: Speckle Noise Reduction
            try:
                denoised_image = self._reduce_speckle_noise(resized_image)
                self._save_stage_output(denoised_image, image_id, PreprocessingStage.DENOISED)
                results['stage_outputs']['denoised'] = self._get_stage_output_path(image_id, PreprocessingStage.DENOISED)
                results['stages_completed'].append('denoise')
                logger.info(f"Speckle noise reduction completed for {image_id}")
            except Exception as e:
                logger.error(f"Speckle noise reduction failed for {image_id}: {str(e)}")
                if not self.config.CONTINUE_ON_ERROR:
                    raise
            
            # Stage 3: CLAHE
            try:
                clahe_image = self._apply_clahe(denoised_image)
                self._save_stage_output(clahe_image, image_id, PreprocessingStage.CLAHE)
                results['stage_outputs']['clahe'] = self._get_stage_output_path(image_id, PreprocessingStage.CLAHE)
                results['stages_completed'].append('clahe')
                logger.info(f"CLAHE completed for {image_id}")
            except Exception as e:
                logger.error(f"CLAHE failed for {image_id}: {str(e)}")
                if not self.config.CONTINUE_ON_ERROR:
                    raise
            
            # Stage 4: U-Net Segmentation
            try:
                segmentation_mask = self._segment_ovarian_follicles(clahe_image)
                self._save_stage_output(segmentation_mask, image_id, PreprocessingStage.SEGMENTATION)
                results['stage_outputs']['segmentation'] = self._get_stage_output_path(image_id, PreprocessingStage.SEGMENTATION)
                results['stages_completed'].append('segmentation')
                logger.info(f"Segmentation completed for {image_id}")
            except Exception as e:
                logger.error(f"Segmentation failed for {image_id}: {str(e)}")
                if not self.config.CONTINUE_ON_ERROR:
                    raise
            
            # Stage 5: ROI Extraction
            try:
                roi_image = self._extract_roi(clahe_image, segmentation_mask)
                self._save_stage_output(roi_image, image_id, PreprocessingStage.ROI)
                results['stage_outputs']['roi'] = self._get_stage_output_path(image_id, PreprocessingStage.ROI)
                results['stages_completed'].append('roi_extraction')
                logger.info(f"ROI extraction completed for {image_id}")
            except Exception as e:
                logger.error(f"ROI extraction failed for {image_id}: {str(e)}")
                if not self.config.CONTINUE_ON_ERROR:
                    raise
            
            # Stage 6: Normalization
            try:
                normalized_image = self._normalize_image(roi_image)
                self._save_stage_output(normalized_image, image_id, PreprocessingStage.NORMALIZED)
                results['stage_outputs']['normalized'] = self._get_stage_output_path(image_id, PreprocessingStage.NORMALIZED)
                results['stages_completed'].append('normalization')
                results['final_image'] = normalized_image
                logger.info(f"Normalization completed for {image_id}")
            except Exception as e:
                logger.error(f"Normalization failed for {image_id}: {str(e)}")
                if not self.config.CONTINUE_ON_ERROR:
                    raise
            
            results['preprocessing_status'] = 'completed'
            logger.info(f"Preprocessing completed successfully for {image_id}")
            
            return results
            
        except Exception as e:
            logger.error(f"Preprocessing failed for {image_id}: {str(e)}")
            results['preprocessing_status'] = 'failed'
            results['error'] = str(e)
            return results
    
    def _load_image(self, image_data: bytes) -> Optional[np.ndarray]:
        """
        Load image from bytes
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Image as numpy array or None if loading fails
        """
        try:
            image_stream = io.BytesIO(image_data)
            pil_image = Image.open(image_stream)
            
            # Convert to RGB if necessary
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Convert to numpy array
            image = np.array(pil_image)
            
            # Convert RGB to BGR for OpenCV compatibility
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            return image
            
        except Exception as e:
            logger.error(f"Failed to load image: {str(e)}")
            return None
    
    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to target resolution
        
        Args:
            image: Input image
            
        Returns:
            Resized image
        """
        params = self.config.get_resize_params()
        target_width = params['target_width']
        target_height = params['target_height']
        maintain_aspect_ratio = params['maintain_aspect_ratio']
        
        # Map interpolation method name to OpenCV constant
        interpolation_map = {
            'linear': cv2.INTER_LINEAR,
            'cubic': cv2.INTER_CUBIC,
            'nearest': cv2.INTER_NEAREST,
            'lanczos': cv2.INTER_LANCZOS4
        }
        interpolation = interpolation_map.get(params['interpolation'], cv2.INTER_LINEAR)
        
        if maintain_aspect_ratio:
            # Calculate scaling factor to fit within target dimensions
            h, w = image.shape[:2]
            scale = min(target_width / w, target_height / h)
            
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Resize
            resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
            
            # Pad to target size
            pad_h = target_height - new_h
            pad_w = target_width - new_w
            
            # Pad evenly on both sides
            top = pad_h // 2
            bottom = pad_h - top
            left = pad_w // 2
            right = pad_w - left
            
            padded = cv2.copyMakeBorder(
                resized, top, bottom, left, right,
                cv2.BORDER_CONSTANT,
                value=params['padding_color']
            )
            
            return padded
        else:
            # Force exact dimensions
            return cv2.resize(image, (target_width, target_height), interpolation=interpolation)
    
    def _reduce_speckle_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Apply speckle noise reduction using Lee filter or SRAD
        
        Args:
            image: Input image
            
        Returns:
            Denoised image
        """
        params = self.config.get_noise_reduction_params()
        method = params['method']
        
        if method == 'lee':
            return self._apply_lee_filter(image, params)
        elif method == 'srad':
            return self._apply_srad(image, params)
        else:
            logger.warning(f"Unknown noise reduction method: {method}, using Lee filter")
            return self._apply_lee_filter(image, params)
    
    def _apply_lee_filter(self, image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """
        Apply Lee filter for speckle noise reduction
        
        Args:
            image: Input image
            params: Lee filter parameters
            
        Returns:
            Denoised image
        """
        # Convert to grayscale for filtering
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        filter_size = params['filter_size']
        iterations = params['iterations']
        
        denoised = gray.copy()
        
        for _ in range(iterations):
            # Apply Lee filter using adaptive Wiener filter approximation
            # Lee filter is a special case of adaptive filtering
            denoised = cv2.fastNlMeansDenoising(
                denoised, 
                None, 
                h=10,  # Filter strength
                templateWindowSize=filter_size,
                searchWindowSize=21
            )
        
        # Convert back to original color space if needed
        if len(image.shape) == 3:
            denoised = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
        
        return denoised
    
    def _apply_srad(self, image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """
        Apply Speckle Reducing Anisotropic Diffusion (SRAD)
        
        SRAD is specifically designed for ultrasound speckle noise reduction.
        It uses the Instantaneous Coefficient of Variation (ICOV) which is
        speckle-aware, unlike standard anisotropic diffusion that uses gradient magnitude.
        
        The diffusion coefficient is based on ICOV/q0 where:
        - ICOV (q) = sqrt(local_variance) / local_mean
        - q0 is the speckle scale parameter estimated from homogeneous regions
        
        Mathematical Formulation (Yu & Acton, IEEE TIP 2002):
        - Diffusion equation: dI/dt = div(c(q) * grad(I))
        - Diffusion coefficient: c(q) = 1 / (1 + max(0, q^2 - q0^2) / (q0^2 * (1 + q0^2)))
        - Bounds: c(q) in [0.0, 1.0], where c=1 in homogeneous regions (q <= q0)
          and c -> 0 at anatomical edges (q > q0).
        - Discrete divergence: div = c_N*d_N + c_S*d_S + c_W*d_W + c_E*d_E
          where d_N = I(i-1,j) - I(i,j), d_S = I(i+1,j) - I(i,j), etc.
        
        Args:
            image: Input image
            params: SRAD parameters including iterations, time_step, q0, kernel_size
            
        Returns:
            Denoised image
        """
        try:
            # Convert to grayscale for SRAD processing
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # SRAD parameters
            iterations = params.get('iterations', self.config.SRAD_ITERATIONS)
            delta_t = params.get('time_step', self.config.SRAD_TIME_STEP)
            q0_config = params.get('q0', None)
            kernel_size = params.get('kernel_size', self.config.SRAD_KERNEL_SIZE)
            q0_estimation_region = params.get('q0_estimation_region', 'center')
            
            # Ensure kernel size is odd
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            # Convert to float64 for numerical precision during diffusion
            denoised64 = gray.astype(np.float64)
            
            # Estimate q0 from image if not provided
            if q0_config is None:
                q0 = self._estimate_q0(denoised64, kernel_size, q0_estimation_region)
                logger.info(f"Auto-estimated q0: {q0:.4f}")
            else:
                q0 = float(q0_config)
                logger.info(f"Using configured q0: {q0:.4f}")
            
            # Create kernel for local statistics
            kernel = np.ones((kernel_size, kernel_size), np.float64) / (kernel_size * kernel_size)
            
            # SRAD iterations - perform all iterations in floating point
            for iteration in range(iterations):
                # 1. Calculate local mean using convolution with reflection border
                local_mean = cv2.filter2D(denoised64, -1, kernel, borderType=cv2.BORDER_REFLECT)
                
                # 2. Calculate local variance: E[X^2] - (E[X])^2
                local_squared_mean = cv2.filter2D(denoised64**2, -1, kernel, borderType=cv2.BORDER_REFLECT)
                local_variance = np.maximum(local_squared_mean - local_mean**2, 0.0)
                
                # 3. Calculate ICOV (Instantaneous Coefficient of Variation): q = sqrt(variance) / mean
                local_std = np.sqrt(local_variance)
                q = np.divide(local_std, local_mean, 
                              out=np.zeros_like(local_std), 
                              where=local_mean > 1e-6)
                
                # 4. Calculate SRAD diffusion coefficient c(q) [Yu & Acton 2002, Eq. 33]
                # Homogeneous regions (q <= q0): max(0, q^2 - q0^2) = 0 -> c(q) = 1.0 (isotropic diffusion).
                # Edge regions (q > q0): c(q) decreases toward 0.0 (inhibits diffusion across edges).
                num = np.maximum(0.0, q**2 - q0**2)
                den = q0**2 * (1.0 + q0**2)
                diffusion_coeff = 1.0 / (1.0 + np.divide(num, den, 
                                                         out=np.zeros_like(num), 
                                                         where=den > 1e-6))
                diffusion_coeff = np.clip(diffusion_coeff, 0.0, 1.0)
                
                # 5. Compute directional neighbor gradients with reflection boundary padding
                # Reflection padding ensures Neumann (zero-flux) boundary conditions at image borders
                I_pad = np.pad(denoised64, ((1, 1), (1, 1)), mode='reflect')
                c_pad = np.pad(diffusion_coeff, ((1, 1), (1, 1)), mode='reflect')
                
                # Gradients to 4-connected neighbors: d = I_neighbor - I_center
                d_n = I_pad[:-2, 1:-1] - denoised64   # North (i-1, j)
                d_s = I_pad[2:, 1:-1] - denoised64    # South (i+1, j)
                d_w = I_pad[1:-1, :-2] - denoised64   # West (i, j-1)
                d_e = I_pad[1:-1, 2:] - denoised64    # East (i, j+1)
                
                # Directional conduction coefficients (averaged between center and neighbor)
                c_n = 0.5 * (diffusion_coeff + c_pad[:-2, 1:-1])
                c_s = 0.5 * (diffusion_coeff + c_pad[2:, 1:-1])
                c_w = 0.5 * (diffusion_coeff + c_pad[1:-1, :-2])
                c_e = 0.5 * (diffusion_coeff + c_pad[1:-1, 2:])
                
                # 6. Calculate discrete divergence of diffusion flux
                div_term = c_n * d_n + c_s * d_s + c_w * d_w + c_e * d_e
                
                # 7. Apply SRAD update: I^{k+1} = I^k + dt * div(c * grad(I))
                denoised64 = denoised64 + delta_t * div_term
            
            # Final clipping to valid range after all floating point iterations complete
            denoised64 = np.clip(denoised64, 0, 255)
            
            # Convert back to uint8
            denoised = denoised64.astype(np.uint8)
            
            # Convert back to original color space if needed
            if len(image.shape) == 3:
                denoised_bgr = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
                return denoised_bgr
            else:
                return denoised
            
        except Exception as e:
            logger.error(f"SRAD failed: {str(e)}")
            logger.warning("Falling back to Lee filter")
            return self._apply_lee_filter(image, self.config.get_noise_reduction_params())
    
    def _estimate_q0(self, image: np.ndarray, kernel_size: int, region: str = 'center') -> float:
        """
        Estimate q0 (speckle scale parameter) from homogeneous regions of the image.
        
        q0 represents the coefficient of variation in fully developed speckle regions.
        For ultrasound images, this is typically estimated from homogeneous areas.
        
        Args:
            image: Input image (float64)
            kernel_size: Kernel size for local statistics
            region: Region to use for estimation ('center', 'corners', 'full')
            
        Returns:
            Estimated q0 value
        """
        h, w = image.shape
        
        # Select region for q0 estimation
        if region == 'center':
            # Use center 50% of image
            margin_h = h // 4
            margin_w = w // 4
            region_img = image[margin_h:h-margin_h, margin_w:w-margin_w]
        elif region == 'corners':
            # Use corners (each 25% of image)
            corner_size_h = h // 4
            corner_size_w = w // 4
            corners = np.concatenate([
                image[:corner_size_h, :corner_size_w].flatten(),
                image[:corner_size_h, -corner_size_w:].flatten(),
                image[-corner_size_h:, :corner_size_w].flatten(),
                image[-corner_size_h:, -corner_size_w:].flatten()
            ])
            region_img = corners.reshape(-1, 1)
        else:  # 'full'
            region_img = image
        
        # Calculate local statistics in the selected region
        kernel = np.ones((kernel_size, kernel_size), np.float64) / (kernel_size * kernel_size)
        
        if len(region_img.shape) == 1:
            # For corner case, use simple statistics
            region_mean = np.mean(region_img)
            region_std = np.std(region_img)
            q0_estimated = region_std / (region_mean + 1e-6)
        else:
            # For 2D regions, use local statistics
            local_mean = cv2.filter2D(region_img, -1, kernel, borderType=cv2.BORDER_REFLECT)
            local_squared_mean = cv2.filter2D(region_img**2, -1, kernel, borderType=cv2.BORDER_REFLECT)
            local_variance = np.maximum(local_squared_mean - local_mean**2, 0.0)
            local_std = np.sqrt(local_variance)
            
            # Calculate coefficient of variation
            icov = np.divide(local_std, local_mean, 
                            out=np.zeros_like(local_std), 
                            where=local_mean > 1e-6)
            
            # Use median of ICOV as q0 estimate (robust to outliers)
            q0_estimated = np.median(icov)
        
        # Clamp to reasonable range for ultrasound images
        q0_estimated = np.clip(q0_estimated, 0.05, 1.0)
        
        return float(q0_estimated)
    
    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
        
        Args:
            image: Input image
            
        Returns:
            CLAHE-enhanced image
        """
        params = self.config.get_clahe_params()
        
        # Convert to LAB color space for better results
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0]
        else:
            l_channel = image.copy()
        
        # Create CLAHE object
        clahe = cv2.createCLAHE(
            clipLimit=params['clip_limit'],
            tileGridSize=params['tile_size']
        )
        
        # Apply CLAHE to L channel
        l_channel_clahe = clahe.apply(l_channel)
        
        # Merge back if color image
        if len(image.shape) == 3:
            lab[:, :, 0] = l_channel_clahe
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            enhanced = l_channel_clahe
        
        return enhanced
    
    def _segment_ovarian_follicles(self, image: np.ndarray) -> np.ndarray:
        """
        Segment ovarian follicles using U-Net model
        
        Args:
            image: Input image
            
        Returns:
            Binary segmentation mask
        """
        # For now, use a simple threshold-based segmentation
        # In production, this would use a trained U-Net model
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Apply morphological operations to clean up
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Note: This is a placeholder. Replace with actual U-Net model:
        # if self.unet_model is None:
        #     self._load_unet_model()
        # mask = self.unet_model.predict(image[np.newaxis, ...])
        
        return binary
    
    def _extract_roi(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Extract Region of Interest (ROI) from image using segmentation mask
        
        Args:
            image: Input image
            mask: Binary segmentation mask
            
        Returns:
            Bounding box cropped ROI
        """
        # Find contours in mask
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            logger.warning("No contours found in segmentation mask, returning full image")
            return image
        
        # Find the largest contour (assuming it's the ovarian follicle)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Check if ROI size is within acceptable range
        area = w * h
        if area < self.config.MIN_ROI_AREA:
            logger.warning(f"ROI area {area} below minimum {self.config.MIN_ROI_AREA}")
            return image
        if area > self.config.MAX_ROI_AREA:
            logger.warning(f"ROI area {area} above maximum {self.config.MAX_ROI_AREA}")
            return image
        
        # Extract ROI with some padding
        padding = 10
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)
        
        roi = image[y:y+h, x:x+w]
        
        return roi
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image pixel values
        
        Args:
            image: Input image
            
        Returns:
            Normalized image
        """
        params = self.config.get_normalization_params()
        method = params['method']
        
        # Convert to float for normalization
        if image.dtype != np.float32:
            image_float = image.astype(np.float32)
        else:
            image_float = image.copy()
        
        if method == 'minmax':
            # Min-max normalization
            min_val = np.min(image_float)
            max_val = np.max(image_float)
            if max_val - min_val > 0:
                normalized = (image_float - min_val) / (max_val - min_val)
                normalized = normalized * (params['max'] - params['min']) + params['min']
            else:
                normalized = np.full_like(image_float, params['min'])
        
        elif method == 'zscore':
            # Z-score normalization
            mean = np.mean(image_float)
            std = np.std(image_float)
            if std > 0:
                normalized = (image_float - mean) / std
                normalized = normalized * params['z_score_std'] + params['z_score_mean']
            else:
                normalized = image_float - mean
        
        elif method == 'percentile':
            # Percentile normalization
            lower = np.percentile(image_float, params['percentile_lower'])
            upper = np.percentile(image_float, params['percentile_upper'])
            if upper - lower > 0:
                normalized = np.clip(image_float, lower, upper)
                normalized = (normalized - lower) / (upper - lower)
                normalized = normalized * (params['max'] - params['min']) + params['min']
            else:
                normalized = np.full_like(image_float, params['min'])
        
        else:
            logger.warning(f"Unknown normalization method: {method}, using minmax")
            return self._normalize_image_with_method(image_float, 'minmax', params)
        
        # Clip to valid range
        normalized = np.clip(normalized, 0, 255)
        
        # Convert back to uint8
        return normalized.astype(np.uint8)
    
    def _normalize_image_with_method(
        self, 
        image: np.ndarray, 
        method: str, 
        params: Dict[str, Any]
    ) -> np.ndarray:
        """Helper method for normalization"""
        # Recursive call to avoid code duplication
        original_method = self.config.NORMALIZATION_METHOD
        self.config.NORMALIZATION_METHOD = method
        result = self._normalize_image(image)
        self.config.NORMALIZATION_METHOD = original_method
        return result
    
    def _save_stage_output(
        self, 
        image: np.ndarray, 
        image_id: str, 
        stage: PreprocessingStage
    ) -> None:
        """
        Save output of a preprocessing stage
        
        Args:
            image: Image to save
            image_id: Image identifier
            stage: Preprocessing stage
        """
        output_path = self._get_stage_output_path(image_id, stage)
        
        # Convert BGR to RGB for saving
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Save image
        pil_image = Image.fromarray(image_rgb)
        pil_image.save(output_path, quality=self.config.OUTPUT_QUALITY)
        
        logger.debug(f"Saved stage output to {output_path}")
    
    def _get_stage_output_path(self, image_id: str, stage: PreprocessingStage) -> str:
        """
        Get output path for a preprocessing stage
        
        Args:
            image_id: Image identifier
            stage: Preprocessing stage
            
        Returns:
            Full path to output file
        """
        return self.config.get_output_path(stage.value, image_id)
    
    def _load_unet_model(self) -> None:
        """
        Load U-Net model for segmentation
        This is a placeholder for loading a pre-trained model
        """
        # In production, load actual U-Net model:
        # from tensorflow.keras.models import load_model
        # self.unet_model = load_model(self.config.UNET_MODEL_PATH)
        logger.info("U-Net model loading placeholder - using threshold-based segmentation")
