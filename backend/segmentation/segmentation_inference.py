"""
Segmentation Inference Module
Handles U-Net model loading, inference, and ROI extraction
"""

import os
import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

from models.unet import UNet
from configs.preprocessing_config import PreprocessingConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SegmentationInference:
    """Handles U-Net segmentation inference and ROI extraction"""
    
    def __init__(
        self,
        weights_path: Optional[str] = None,
        config: Optional[PreprocessingConfig] = None
    ):
        """
        Initialize segmentation inference
        
        Args:
            weights_path: Path to trained U-Net weights file (.h5 or .keras)
            config: Preprocessing configuration
        """
        self.config = config or PreprocessingConfig()
        self.weights_path = weights_path or self.config.UNET_MODEL_PATH
        self.model = None
        self.model_loaded = False
        
        # Try to load model if weights path is provided
        if self.weights_path and os.path.exists(self.weights_path):
            self.load_model()
        else:
            logger.warning(
                f"No trained U-Net weights found at {self.weights_path}. "
                "Segmentation will not be available. "
                "Please provide a trained model weights file."
            )
    
    def load_model(self) -> bool:
        """
        Load U-Net model with trained weights
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            if not self.weights_path:
                logger.error("No weights path provided")
                return False
            
            if not os.path.exists(self.weights_path):
                logger.error(f"Weights file not found: {self.weights_path}")
                return False
            
            # Create U-Net architecture
            unet = UNet(
                input_size=self.config.UNET_INPUT_SIZE,
                num_classes=self.config.UNET_NUM_CLASSES,
                filters=self.config.UNET_FILTERS,
                depth=self.config.UNET_DEPTH,
                dropout_rate=self.config.UNET_DROPOUT_RATE
            )
            
            # Build model
            self.model = unet.build_model()
            
            # Load weights
            self.model.load_weights(self.weights_path)
            
            self.model_loaded = True
            logger.info(f"U-Net model loaded successfully from {self.weights_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load U-Net model: {str(e)}")
            self.model_loaded = False
            return False
    
    def segment_image(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Perform ovarian follicle segmentation on image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Tuple of (segmentation_mask, metadata_dict)
            If model not loaded, returns (None, error_metadata)
        """
        if not self.model_loaded:
            logger.error("Cannot perform segmentation: model not loaded")
            return None, {
                'status': 'failed',
                'error': 'Model not loaded. Please provide trained U-Net weights.',
                'weights_path': self.weights_path
            }
        
        try:
            # Preprocess image for model
            preprocessed = self._preprocess_for_model(image)
            
            # Perform inference
            prediction = self.model.predict(preprocessed, verbose=0)
            
            # Postprocess prediction
            mask = self._postprocess_prediction(prediction)
            
            metadata = {
                'status': 'success',
                'model_loaded': True,
                'weights_path': self.weights_path,
                'input_shape': image.shape,
                'output_shape': mask.shape
            }
            
            return mask, metadata
            
        except Exception as e:
            logger.error(f"Segmentation failed: {str(e)}")
            return None, {
                'status': 'failed',
                'error': str(e),
                'model_loaded': self.model_loaded
            }
    
    def _preprocess_for_model(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for U-Net model
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Preprocessed image ready for model input
        """
        # Resize to model input size
        target_size = self.config.UNET_INPUT_SIZE
        resized = cv2.resize(image, (target_size[1], target_size[0]))
        
        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        preprocessed = np.expand_dims(normalized, axis=0)
        
        return preprocessed
    
    def _postprocess_prediction(self, prediction: np.ndarray) -> np.ndarray:
        """
        Postprocess model prediction to get binary mask
        
        Args:
            prediction: Model output
            
        Returns:
            Binary segmentation mask
        """
        # Remove batch dimension
        prediction = prediction[0]
        
        # Apply threshold
        threshold = self.config.SEGMENTATION_THRESHOLD
        binary_mask = (prediction > threshold).astype(np.uint8) * 255
        
        # Squeeze if needed
        if len(binary_mask.shape) == 3 and binary_mask.shape[2] == 1:
            binary_mask = binary_mask.squeeze(axis=2)
        
        return binary_mask
    
    def extract_roi(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Extract ROI from image using segmentation mask
        
        Args:
            image: Original image
            mask: Binary segmentation mask
            
        Returns:
            Tuple of (roi_image, roi_metadata)
        """
        try:
            # Find contours in mask
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            if not contours or len(contours) == 0:
                logger.warning("No contours found in segmentation mask")
                return image, {
                    'status': 'no_contours',
                    'bbox': None,
                    'area': 0,
                    'roi_used_full_image': True
                }
            
            # Find the largest contour (assuming it's the ovarian follicle)
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            # Check if ROI size is within acceptable range
            if area < self.config.MIN_ROI_AREA:
                logger.warning(
                    f"ROI area {area} below minimum {self.config.MIN_ROI_AREA}, "
                    "using full image"
                )
                return image, {
                    'status': 'roi_too_small',
                    'bbox': None,
                    'area': area,
                    'roi_used_full_image': True
                }
            
            if area > self.config.MAX_ROI_AREA:
                logger.warning(
                    f"ROI area {area} above maximum {self.config.MAX_ROI_AREA}, "
                    "using full image"
                )
                return image, {
                    'status': 'roi_too_large',
                    'bbox': None,
                    'area': area,
                    'roi_used_full_image': True
                }
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Extract ROI with padding
            padding = 10
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2 * padding)
            h = min(image.shape[0] - y, h + 2 * padding)
            
            roi = image[y:y+h, x:x+w]
            
            bbox = {
                'x': int(x),
                'y': int(y),
                'width': int(w),
                'height': int(h)
            }
            
            metadata = {
                'status': 'success',
                'bbox': bbox,
                'area': int(area),
                'roi_used_full_image': False,
                'contour_count': len(contours)
            }
            
            return roi, metadata
            
        except Exception as e:
            logger.error(f"ROI extraction failed: {str(e)}")
            return image, {
                'status': 'failed',
                'error': str(e),
                'bbox': None,
                'area': 0,
                'roi_used_full_image': True
            }
    
    def create_overlay(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Create overlay visualization of image and mask
        
        Args:
            image: Original image
            mask: Binary segmentation mask
            alpha: Transparency for overlay (0-1)
            
        Returns:
            Overlay image
        """
        try:
            # Ensure mask is same size as image
            if mask.shape[:2] != image.shape[:2]:
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
            
            # Create colored mask (green for segmentation)
            colored_mask = np.zeros_like(image)
            colored_mask[mask > 0] = [0, 255, 0]  # Green
            
            # Blend image and mask
            overlay = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
            
            return overlay
            
        except Exception as e:
            logger.error(f"Overlay creation failed: {str(e)}")
            return image
    
    def normalize_roi(self, roi: np.ndarray) -> np.ndarray:
        """
        Normalize ROI image
        
        Args:
            roi: ROI image
            
        Returns:
            Normalized ROI
        """
        params = self.config.get_normalization_params()
        method = params['method']
        
        # Convert to float for normalization
        if roi.dtype != np.float32:
            roi_float = roi.astype(np.float32)
        else:
            roi_float = roi.copy()
        
        if method == 'minmax':
            min_val = np.min(roi_float)
            max_val = np.max(roi_float)
            if max_val - min_val > 0:
                normalized = (roi_float - min_val) / (max_val - min_val)
                normalized = normalized * (params['max'] - params['min']) + params['min']
            else:
                normalized = np.full_like(roi_float, params['min'])
        
        elif method == 'zscore':
            mean = np.mean(roi_float)
            std = np.std(roi_float)
            if std > 0:
                normalized = (roi_float - mean) / std
                normalized = normalized * params['z_score_std'] + params['z_score_mean']
            else:
                normalized = roi_float - mean
        
        elif method == 'percentile':
            lower = np.percentile(roi_float, params['percentile_lower'])
            upper = np.percentile(roi_float, params['percentile_upper'])
            if upper - lower > 0:
                normalized = np.clip(roi_float, lower, upper)
                normalized = (normalized - lower) / (upper - lower)
                normalized = normalized * (params['max'] - params['min']) + params['min']
            else:
                normalized = np.full_like(roi_float, params['min'])
        
        else:
            logger.warning(f"Unknown normalization method: {method}, using minmax")
            return self._normalize_with_method(roi_float, 'minmax', params)
        
        # Clip to valid range
        normalized = np.clip(normalized, 0, 255)
        
        # Convert back to uint8
        return normalized.astype(np.uint8)
    
    def _normalize_with_method(
        self,
        image: np.ndarray,
        method: str,
        params: Dict[str, Any]
    ) -> np.ndarray:
        """Helper method for normalization"""
        original_method = self.config.NORMALIZATION_METHOD
        self.config.NORMALIZATION_METHOD = method
        result = self.normalize_roi(image)
        self.config.NORMALIZATION_METHOD = original_method
        return result
    
    def save_segmentation_outputs(
        self,
        image_id: str,
        image: np.ndarray,
        mask: Optional[np.ndarray],
        roi: np.ndarray,
        overlay: Optional[np.ndarray],
        bbox_info: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Save all segmentation outputs
        
        Args:
            image_id: Image identifier
            image: Original image
            mask: Segmentation mask (optional)
            roi: Extracted ROI
            overlay: Overlay visualization (optional)
            bbox_info: Bounding box information
            metadata: Segmentation metadata
            
        Returns:
            Dictionary of output file paths
        """
        output_paths = {}
        
        try:
            # Ensure output directories exist
            self.config.ensure_output_directories()
            
            # Save original image
            original_path = self.config.get_output_path('original', image_id)
            self._save_image(image, original_path)
            output_paths['original'] = original_path
            
            # Save segmentation mask if available
            if mask is not None:
                mask_path = self.config.get_output_path('segmentation', image_id)
                self._save_image(mask, mask_path)
                output_paths['mask'] = mask_path
            
            # Save ROI
            roi_path = self.config.get_output_path('roi', image_id)
            self._save_image(roi, roi_path)
            output_paths['roi'] = roi_path
            
            # Save overlay if available
            if overlay is not None:
                overlay_path = os.path.join(
                    self.config.OUTPUT_BASE_DIR,
                    'overlay',
                    f"{image_id}.{self.config.OUTPUT_FORMAT}"
                )
                os.makedirs(os.path.dirname(overlay_path), exist_ok=True)
                self._save_image(overlay, overlay_path)
                output_paths['overlay'] = overlay_path
            
            # Save normalized ROI
            normalized_roi = self.normalize_roi(roi)
            normalized_path = self.config.get_output_path('normalized', image_id)
            self._save_image(normalized_roi, normalized_path)
            output_paths['normalized'] = normalized_path
            
            # Save bbox info as JSON
            import json
            bbox_path = os.path.join(
                self.config.OUTPUT_BASE_DIR,
                'bbox_info',
                f"{image_id}.json"
            )
            os.makedirs(os.path.dirname(bbox_path), exist_ok=True)
            with open(bbox_path, 'w') as f:
                json.dump({
                    'bbox': bbox_info,
                    'metadata': metadata
                }, f, indent=2)
            output_paths['bbox_info'] = bbox_path
            
            logger.info(f"Saved segmentation outputs for {image_id}")
            
        except Exception as e:
            logger.error(f"Failed to save segmentation outputs: {str(e)}")
        
        return output_paths
    
    def _save_image(self, image: np.ndarray, path: str) -> None:
        """
        Save image to file
        
        Args:
            image: Image to save
            path: Output path
        """
        # Convert BGR to RGB for saving
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        pil_image = Image.fromarray(image_rgb)
        pil_image.save(path, quality=self.config.OUTPUT_QUALITY)
