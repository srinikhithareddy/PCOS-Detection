"""
Ultrasound Image Quality Assessment
Implements OpenCV-based image quality metrics (blur, sharpness, brightness, contrast)
"""

import os
import io
from typing import Dict, Any, Optional, Tuple
from enum import Enum
import numpy as np
from PIL import Image
import cv2
import logging

from configs.quality_config import QualityConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QualityCategory(Enum):
    """Image quality categories"""
    GOOD = "good"
    POOR = "poor"
    UNUSABLE = "unusable"


class ProcessingDecision(Enum):
    """Processing decisions based on quality"""
    CONTINUE = "continue"  # Good quality, proceed with preprocessing
    ENHANCE = "enhance"  # Poor quality, mark for enhancement then continue
    FLAG = "flag"  # Extremely corrupted, flag for review


class ImageQualityAssessor:
    """Assesses ultrasound image quality using OpenCV-based metrics"""
    
    def __init__(self, config: Optional[QualityConfig] = None):
        """Initialize the quality assessor"""
        self.config = config or QualityConfig()
        logger.info("ImageQualityAssessor initialized")
    
    def assess_image_quality(
        self,
        image_data: bytes,
        image_id: str,
        filename: str
    ) -> Dict[str, Any]:
        """
        Perform comprehensive image quality assessment
        
        Args:
            image_data: Raw image bytes
            image_id: Unique identifier for the image
            filename: Original filename
            
        Returns:
            Dictionary containing quality assessment results
        """
        try:
            # Step 1: Basic validation (file format, size, readability)
            basic_validation = self._validate_basic_image_properties(image_data, filename)
            if not basic_validation['valid']:
                return self._create_unusable_report(
                    image_id, filename, 
                    reason=basic_validation['reason']
                )
            
            # Step 2: Load image
            image = self._load_image(image_data)
            if image is None:
                return self._create_unusable_report(
                    image_id, filename,
                    reason="Failed to load image - possibly corrupted"
                )
            
            # Step 3: Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(image)
            
            # Step 4: Determine quality category and decision
            quality_category, processing_decision = self._determine_quality_decision(
                quality_metrics
            )
            
            # Step 5: Generate quality report
            report = self._create_quality_report(
                image_id=image_id,
                filename=filename,
                quality_metrics=quality_metrics,
                quality_category=quality_category,
                processing_decision=processing_decision,
                image_dimensions=image.shape[:2]
            )
            
            logger.info(f"Quality assessment completed for {filename}: "
                       f"{quality_category.value} - {processing_decision.value}")
            
            return report
            
        except Exception as e:
            logger.error(f"Error during quality assessment for {filename}: {str(e)}")
            return self._create_unusable_report(
                image_id, filename,
                reason=f"Assessment error: {str(e)}"
            )
    
    def _validate_basic_image_properties(
        self, 
        image_data: bytes, 
        filename: str
    ) -> Dict[str, Any]:
        """
        Validate basic image properties
        
        Args:
            image_data: Raw image bytes
            filename: Original filename
            
        Returns:
            Dictionary with validation result and reason if invalid
        """
        result = {'valid': True, 'reason': None}
        
        # Check file size
        if len(image_data) > self.config.MAX_IMAGE_SIZE_BYTES:
            result['valid'] = False
            result['reason'] = f"File size exceeds maximum allowed size"
            return result
        
        # Check file extension
        ext = filename.lower().split('.')[-1] if '.' in filename else ''
        if ext not in self.config.ALLOWED_IMAGE_FORMATS:
            result['valid'] = False
            result['reason'] = f"Unsupported file format: {ext}"
            return result
        
        # Try to read image header to check for corruption
        try:
            img = Image.open(io.BytesIO(image_data))
            img.verify()  # Verify without loading
        except Exception as e:
            result['valid'] = False
            result['reason'] = f"Image file corrupted or unreadable: {str(e)}"
            return result
        
        return result
    
    def _load_image(self, image_data: bytes) -> Optional[np.ndarray]:
        """
        Load image from bytes
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Image as numpy array or None if loading fails
        """
        try:
            # Reset bytes stream
            image_stream = io.BytesIO(image_data)
            
            # Load with PIL first for better error handling
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
    
    def _calculate_quality_metrics(self, image: np.ndarray) -> Dict[str, float]:
        """
        Calculate image quality metrics using OpenCV
        
        Args:
            image: Image as numpy array (BGR format)
            
        Returns:
            Dictionary containing quality metrics
        """
        metrics = {}
        
        try:
            # Convert to grayscale for some metrics
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calculate Laplacian variance (blur detection - higher is sharper)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            metrics['sharpness'] = float(laplacian_var)
            
            # Calculate brightness metrics
            metrics['mean_brightness'] = float(np.mean(gray))
            metrics['std_brightness'] = float(np.std(gray))
            
            # Calculate contrast
            metrics['contrast'] = float(np.std(gray) / (np.mean(gray) + 1e-6))
            
            # Calculate overall quality score (normalized)
            # Combine sharpness, brightness, and contrast into a single score
            # Higher sharpness = better quality
            # Moderate brightness (50-150) = better quality
            # Higher contrast = better quality
            
            sharpness_score = min(laplacian_var / 1000.0, 1.0)  # Normalize to 0-1
            brightness_score = 1.0 - abs(np.mean(gray) - 100) / 100.0  # Optimal around 100
            brightness_score = max(0.0, brightness_score)
            contrast_score = min(np.std(gray) / 100.0, 1.0)  # Normalize to 0-1
            
            # Combined quality score (0-100, higher is better)
            combined_score = (sharpness_score * 0.5 + brightness_score * 0.3 + contrast_score * 0.2) * 100
            metrics['quality_score'] = float(combined_score)
            
        except Exception as e:
            logger.warning(f"Quality metrics calculation failed: {str(e)}")
            metrics['sharpness'] = None
            metrics['mean_brightness'] = None
            metrics['std_brightness'] = None
            metrics['contrast'] = None
            metrics['quality_score'] = None
        
        return metrics
    
    def _determine_quality_decision(
        self, 
        quality_metrics: Dict[str, float]
    ) -> Tuple[QualityCategory, ProcessingDecision]:
        """
        Determine quality category and processing decision
        
        Args:
            quality_metrics: Dictionary of quality metrics
            
        Returns:
            Tuple of (QualityCategory, ProcessingDecision)
        """
        # Use the combined quality score (0-100, higher is better)
        score = quality_metrics.get('quality_score', 0.0)
        
        # Determine category based on score (higher is better)
        # Good: score >= 60
        # Poor: score >= 30 and < 60
        # Unusable: score < 30
        
        if score >= 60.0:
            quality_category = QualityCategory.GOOD
            processing_decision = ProcessingDecision.CONTINUE
        elif score >= 30.0:
            quality_category = QualityCategory.POOR
            processing_decision = ProcessingDecision.ENHANCE
        else:
            quality_category = QualityCategory.UNUSABLE
            processing_decision = ProcessingDecision.FLAG
        
        return quality_category, processing_decision
    
    def _create_quality_report(
        self,
        image_id: str,
        filename: str,
        quality_metrics: Dict[str, float],
        quality_category: QualityCategory,
        processing_decision: ProcessingDecision,
        image_dimensions: Tuple[int, int]
    ) -> Dict[str, Any]:
        """
        Create comprehensive quality report
        
        Args:
            image_id: Unique image identifier
            filename: Original filename
            quality_metrics: Calculated quality metrics
            quality_category: Determined quality category
            processing_decision: Processing decision
            image_dimensions: Image dimensions (height, width)
            
        Returns:
            Quality report dictionary
        """
        return {
            'image_id': image_id,
            'filename': filename,
            'assessment_timestamp': self._get_timestamp(),
            'image_dimensions': {
                'height': int(image_dimensions[0]),
                'width': int(image_dimensions[1])
            },
            'quality_metrics': quality_metrics,
            'quality_score': quality_metrics.get('quality_score', 0.0),
            'quality_category': quality_category.value,
            'processing_decision': processing_decision.value,
            'requires_enhancement': processing_decision == ProcessingDecision.ENHANCE,
            'flagged_for_review': processing_decision == ProcessingDecision.FLAG,
            'assessment_status': 'completed'
        }
    
    def _create_unusable_report(
        self,
        image_id: str,
        filename: str,
        reason: str
    ) -> Dict[str, Any]:
        """
        Create report for unusable images
        
        Args:
            image_id: Unique image identifier
            filename: Original filename
            reason: Reason for being unusable
            
        Returns:
            Quality report dictionary for unusable image
        """
        return {
            'image_id': image_id,
            'filename': filename,
            'assessment_timestamp': self._get_timestamp(),
            'image_dimensions': None,
            'quality_metrics': {},
            'quality_score': None,
            'quality_category': QualityCategory.UNUSABLE.value,
            'processing_decision': ProcessingDecision.FLAG.value,
            'requires_enhancement': False,
            'flagged_for_review': True,
            'assessment_status': 'failed',
            'failure_reason': reason
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format"""
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()
