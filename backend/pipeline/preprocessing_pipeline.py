"""
End-to-End Preprocessing Pipeline
Integrated pipeline for ultrasound and clinical data preprocessing
"""

import os
import logging
import json
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np
import pandas as pd

from pipeline_config import PipelineConfig
from quality_assessment import ImageQualityAssessor, ClinicalQualityAssessor
from preprocessing import UltrasoundPreprocessor
from clinical_preprocessing import ClinicalPreprocessor
from segmentation import SegmentationInference

# Configure logging
def setup_logging():
    """Setup logging with directory creation"""
    PipelineConfig.ensure_output_directories()
    
    logging.basicConfig(
        level=getattr(logging, PipelineConfig.LOG_LEVEL),
        format=PipelineConfig.LOG_FORMAT,
        handlers=[
            logging.FileHandler(os.path.join(PipelineConfig.OUTPUT_BASE_DIR, PipelineConfig.LOG_FILE)),
            logging.StreamHandler() if PipelineConfig.LOG_TO_CONSOLE else logging.NullHandler()
        ]
    )

setup_logging()
logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """End-to-end preprocessing pipeline for ultrasound and clinical data"""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the preprocessing pipeline
        
        Args:
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()
        self.config.ensure_output_directories()
        
        # Initialize quality assessors
        self.ultrasound_quality_assessor = ImageQualityAssessor()
        self.clinical_quality_assessor = ClinicalQualityAssessor()
        
        # Initialize preprocessors
        self.ultrasound_preprocessor = UltrasoundPreprocessor()
        self.clinical_preprocessor = ClinicalPreprocessor()
        
        # Initialize segmentation inference
        self.segmentation_inference = SegmentationInference()
        
        # Pipeline state
        self.pipeline_start_time = None
        self.pipeline_end_time = None
        
        logger.info("PreprocessingPipeline initialized")
    
    def run_pipeline(
        self,
        patient_id: str,
        ultrasound_image: Optional[bytes] = None,
        ultrasound_filename: Optional[str] = None,
        clinical_data: Optional[Dict[str, Any]] = None,
        mode: str = 'inference'
    ) -> Dict[str, Any]:
        """
        Run the complete end-to-end preprocessing pipeline
        
        Args:
            patient_id: Patient/sample identifier
            ultrasound_image: Raw ultrasound image bytes (optional)
            ultrasound_filename: Original ultrasound filename (optional)
            clinical_data: Clinical features dictionary (optional)
            mode: 'train' or 'inference'
            
        Returns:
            Dictionary containing all preprocessing outputs and metadata
        """
        self.pipeline_start_time = datetime.now()
        logger.info(f"Starting preprocessing pipeline for patient {patient_id}")
        
        # Validate inputs
        if not ultrasound_image and not clinical_data:
            raise ValueError("At least one of ultrasound_image or clinical_data must be provided")
        
        if self.config.REQUIRE_BOTH_MODALITIES and (not ultrasound_image or not clinical_data):
            raise ValueError("Both ultrasound and clinical data are required")
        
        # Initialize results dictionary
        results = {
            'patient_id': patient_id,
            'mode': mode,
            'pipeline_start_time': self.pipeline_start_time.isoformat(),
            'ultrasound_available': ultrasound_image is not None,
            'clinical_available': clinical_data is not None,
            'ultrasound_pipeline': {},
            'clinical_pipeline': {},
            'final_outputs': {},
            'pipeline_status': 'in_progress',
            'errors': []
        }
        
        try:
            # Save raw data
            self._save_raw_data(patient_id, ultrasound_image, clinical_data)
            
            # Run ultrasound pipeline if available
            if ultrasound_image and self.config.RUN_ULTRASOUND_PIPELINE:
                try:
                    results['ultrasound_pipeline'] = self._run_ultrasound_pipeline(
                        patient_id,
                        ultrasound_image,
                        ultrasound_filename,
                        mode
                    )
                except Exception as e:
                    error_msg = f"Ultrasound pipeline failed: {str(e)}"
                    logger.error(error_msg)
                    results['errors'].append(error_msg)
                    if not self.config.CONTINUE_ON_MODALITY_FAILURE:
                        raise
            
            # Run clinical pipeline if available
            if clinical_data and self.config.RUN_CLINICAL_PIPELINE:
                try:
                    results['clinical_pipeline'] = self._run_clinical_pipeline(
                        patient_id,
                        clinical_data,
                        mode
                    )
                except Exception as e:
                    error_msg = f"Clinical pipeline failed: {str(e)}"
                    logger.error(error_msg)
                    results['errors'].append(error_msg)
                    if not self.config.CONTINUE_ON_MODALITY_FAILURE:
                        raise
            
            # Combine final outputs
            results['final_outputs'] = self._combine_final_outputs(
                results['ultrasound_pipeline'],
                results['clinical_pipeline']
            )
            
            # Validate final outputs
            if self.config.VALIDATE_OUTPUTS:
                validation_results = self._validate_outputs(results['final_outputs'])
                results['validation'] = validation_results
            
            # Save preprocessing metadata
            if self.config.SAVE_METADATA:
                self._save_metadata(patient_id, results)
            
            results['pipeline_status'] = 'completed'
            self.pipeline_end_time = datetime.now()
            results['pipeline_end_time'] = self.pipeline_end_time.isoformat()
            results['pipeline_duration_seconds'] = (
                self.pipeline_end_time - self.pipeline_start_time
            ).total_seconds()
            
            logger.info(f"Pipeline completed successfully for patient {patient_id}")
            
        except Exception as e:
            logger.error(f"Pipeline failed for patient {patient_id}: {str(e)}")
            results['pipeline_status'] = 'failed'
            results['error'] = str(e)
            self.pipeline_end_time = datetime.now()
            results['pipeline_end_time'] = self.pipeline_end_time.isoformat()
        
        return results
    
    def _run_ultrasound_pipeline(
        self,
        patient_id: str,
        ultrasound_image: bytes,
        filename: str,
        mode: str
    ) -> Dict[str, Any]:
        """
        Run ultrasound preprocessing pipeline
        
        Args:
            patient_id: Patient identifier
            ultrasound_image: Raw image bytes
            filename: Original filename
            mode: 'train' or 'inference'
            
        Returns:
            Ultrasound pipeline results
        """
        logger.info(f"Starting ultrasound pipeline for patient {patient_id}")
        
        results = {
            'status': 'in_progress',
            'stages_completed': [],
            'quality_assessment': None,
            'preprocessing': None,
            'segmentation': None
        }
        
        # Stage 1: Quality Assessment
        try:
            quality_report = self.ultrasound_quality_assessor.assess_image_quality(
                image_data=ultrasound_image,
                image_id=patient_id,
                filename=filename
            )
            results['quality_assessment'] = quality_report
            results['stages_completed'].append('quality_assessment')
            
            # Check if quality is sufficient
            quality_score = quality_report.get('quality_score', 0)
            if quality_score < self.config.MINIMUM_QUALITY_SCORE:
                logger.warning(
                    f"Ultrasound quality score {quality_score} below threshold "
                    f"{self.config.MINIMUM_QUALITY_SCORE}, skipping preprocessing"
                )
                results['status'] = 'skipped_low_quality'
                return results
            
        except Exception as e:
            logger.error(f"Ultrasound quality assessment failed: {str(e)}")
            results['error'] = str(e)
            results['status'] = 'quality_assessment_failed'
            return results
        
        # Stage 2: Preprocessing (resize, denoise, CLAHE)
        try:
            preprocessing_result = self.ultrasound_preprocessor.preprocess_image(
                image_data=ultrasound_image,
                image_id=patient_id,
                filename=filename
            )
            results['preprocessing'] = preprocessing_result
            results['stages_completed'].append('preprocessing')
            
        except Exception as e:
            logger.error(f"Ultrasound preprocessing failed: {str(e)}")
            results['error'] = str(e)
            results['status'] = 'preprocessing_failed'
            return results
        
        # Stage 3: Segmentation and ROI Extraction
        try:
            # Load the CLAHE-enhanced image for segmentation
            clahe_path = self.ultrasound_preprocessor.config.get_output_path('clahe_enhanced', patient_id)
            if os.path.exists(clahe_path):
                from PIL import Image
                clahe_image = np.array(Image.open(clahe_path))
                
                # Perform segmentation
                segmentation_mask, segmentation_metadata = self.segmentation_inference.segment_image(clahe_image)
                
                if segmentation_mask is not None:
                    # Extract ROI
                    roi, roi_metadata = self.segmentation_inference.extract_roi(clahe_image, segmentation_mask)
                    
                    # Normalize ROI
                    normalized_roi = self.segmentation_inference.normalize_roi(roi)
                    
                    # Save outputs
                    output_paths = self.segmentation_inference.save_segmentation_outputs(
                        image_id=patient_id,
                        image=clahe_image,
                        mask=segmentation_mask,
                        roi=roi,
                        overlay=self.segmentation_inference.create_overlay(clahe_image, segmentation_mask),
                        bbox_info=roi_metadata,
                        metadata=segmentation_metadata
                    )
                    
                    results['segmentation'] = {
                        'status': 'success',
                        'output_paths': output_paths,
                        'roi_metadata': roi_metadata,
                        'segmentation_metadata': segmentation_metadata
                    }
                    results['stages_completed'].append('segmentation')
                else:
                    results['segmentation'] = {
                        'status': 'skipped',
                        'reason': segmentation_metadata.get('error', 'Unknown')
                    }
            else:
                results['segmentation'] = {
                    'status': 'skipped',
                    'reason': 'CLAHE image not found'
                }
                
        except Exception as e:
            logger.error(f"Segmentation failed: {str(e)}")
            results['segmentation'] = {
                'status': 'failed',
                'error': str(e)
            }
        
        results['status'] = 'completed'
        logger.info(f"Ultrasound pipeline completed for patient {patient_id}")
        
        return results
    
    def _run_clinical_pipeline(
        self,
        patient_id: str,
        clinical_data: Dict[str, Any],
        mode: str
    ) -> Dict[str, Any]:
        """
        Run clinical preprocessing pipeline
        
        Args:
            patient_id: Patient identifier
            clinical_data: Clinical features dictionary
            mode: 'train' or 'inference'
            
        Returns:
            Clinical pipeline results
        """
        logger.info(f"Starting clinical pipeline for patient {patient_id}")
        
        results = {
            'status': 'in_progress',
            'stages_completed': [],
            'quality_assessment': None,
            'preprocessing': None
        }
        
        # Stage 1: Quality Assessment
        try:
            quality_report = self.clinical_quality_assessor.assess_clinical_data_quality(
                clinical_data=clinical_data,
                patient_id=patient_id
            )
            results['quality_assessment'] = quality_report
            results['stages_completed'].append('quality_assessment')
            
            # Check reliability score
            reliability_score = quality_report.get('reliability_score', 0)
            if reliability_score < 0.4:  # Unusable threshold
                logger.warning(
                    f"Clinical reliability score {reliability_score} below threshold, "
                    "skipping preprocessing"
                )
                results['status'] = 'skipped_low_reliability'
                return results
            
        except Exception as e:
            logger.error(f"Clinical quality assessment failed: {str(e)}")
            results['error'] = str(e)
            results['status'] = 'quality_assessment_failed'
            return results
        
        # Stage 2: Preprocessing
        try:
            # Check if preprocessor is fitted
            if mode == 'inference' and not self.clinical_preprocessor.is_fitted:
                logger.warning(
                    "Clinical preprocessor not fitted for inference mode. "
                    "Skipping clinical preprocessing."
                )
                results['status'] = 'skipped_not_fitted'
                return results
            
            if mode == 'train':
                feature_vector, preprocessing_metadata = self.clinical_preprocessor.fit_transform(
                    clinical_data,
                    target=None  # Target would be provided for supervised training
                )
            else:
                feature_vector, preprocessing_metadata = self.clinical_preprocessor.transform(
                    clinical_data
                )
            
            results['preprocessing'] = {
                'feature_vector': feature_vector.tolist() if isinstance(feature_vector, np.ndarray) else feature_vector,
                'metadata': preprocessing_metadata
            }
            results['stages_completed'].append('preprocessing')
            
        except Exception as e:
            logger.error(f"Clinical preprocessing failed: {str(e)}")
            results['error'] = str(e)
            results['status'] = 'preprocessing_failed'
            return results
        
        results['status'] = 'completed'
        logger.info(f"Clinical pipeline completed for patient {patient_id}")
        
        return results
    
    def _combine_final_outputs(
        self,
        ultrasound_results: Dict[str, Any],
        clinical_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Combine final outputs from both pipelines
        
        Args:
            ultrasound_results: Ultrasound pipeline results
            clinical_results: Clinical pipeline results
            
        Returns:
            Combined final outputs
        """
        final_outputs = {
            'preprocessed_ultrasound_roi': None,
            'segmentation_mask': None,
            'ultrasound_quality_report': None,
            'clinical_quality_report': None,
            'clinical_reliability_score': None,
            'clinical_feature_vector': None,
            'output_paths': {}
        }
        
        # Extract ultrasound outputs
        if ultrasound_results and ultrasound_results.get('quality_assessment'):
            final_outputs['ultrasound_quality_report'] = ultrasound_results['quality_assessment']
        
        if ultrasound_results and ultrasound_results.get('segmentation'):
            seg = ultrasound_results['segmentation']
            if seg.get('output_paths'):
                final_outputs['output_paths']['segmentation_mask'] = seg['output_paths'].get('mask')
                final_outputs['output_paths']['ultrasound_roi'] = seg['output_paths'].get('roi')
                final_outputs['output_paths']['normalized_roi'] = seg['output_paths'].get('normalized')
                final_outputs['output_paths']['overlay'] = seg['output_paths'].get('overlay')
        
        # Extract clinical outputs
        if clinical_results and clinical_results.get('quality_assessment'):
            final_outputs['clinical_quality_report'] = clinical_results['quality_assessment']
            final_outputs['clinical_reliability_score'] = clinical_results['quality_assessment'].get('reliability_score')
        
        if clinical_results and clinical_results.get('preprocessing'):
            final_outputs['clinical_feature_vector'] = clinical_results['preprocessing'].get('feature_vector')
        
        return final_outputs
    
    def _validate_outputs(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate final outputs
        
        Args:
            outputs: Final outputs dictionary
            
        Returns:
            Validation results
        """
        validation_results = {
            'ultrasound_roi_valid': True,
            'segmentation_mask_valid': True,
            'clinical_vector_valid': True,
            'quality_reports_valid': True,
            'errors': []
        }
        
        # Validate clinical vector
        if outputs.get('clinical_feature_vector'):
            vector = outputs['clinical_feature_vector']
            if not isinstance(vector, (list, np.ndarray)):
                validation_results['clinical_vector_valid'] = False
                validation_results['errors'].append('Clinical vector is not a list or array')
            else:
                length = len(vector)
                if length < self.config.MIN_CLINICAL_VECTOR_LENGTH or length > self.config.MAX_CLINICAL_VECTOR_LENGTH:
                    validation_results['clinical_vector_valid'] = False
                    validation_results['errors'].append(f'Clinical vector length {length} out of range')
        
        # Validate quality reports
        if outputs.get('ultrasound_quality_report'):
            if not isinstance(outputs['ultrasound_quality_report'], dict):
                validation_results['quality_reports_valid'] = False
                validation_results['errors'].append('Ultrasound quality report is not a dict')
        
        if outputs.get('clinical_quality_report'):
            if not isinstance(outputs['clinical_quality_report'], dict):
                validation_results['quality_reports_valid'] = False
                validation_results['errors'].append('Clinical quality report is not a dict')
        
        return validation_results
    
    def _save_raw_data(
        self,
        patient_id: str,
        ultrasound_image: Optional[bytes],
        clinical_data: Optional[Dict[str, Any]]
    ) -> None:
        """
        Save raw input data
        
        Args:
            patient_id: Patient identifier
            ultrasound_image: Raw ultrasound image bytes
            clinical_data: Clinical features dictionary
        """
        # Save ultrasound image
        if ultrasound_image:
            ultrasound_path = self.config.get_output_path('raw_data', patient_id, 'ultrasound.png')
            os.makedirs(os.path.dirname(ultrasound_path), exist_ok=True)
            with open(ultrasound_path, 'wb') as f:
                f.write(ultrasound_image)
            logger.debug(f"Saved raw ultrasound to {ultrasound_path}")
        
        # Save clinical data
        if clinical_data:
            clinical_path = self.config.get_output_path('raw_data', patient_id, 'clinical.json')
            os.makedirs(os.path.dirname(clinical_path), exist_ok=True)
            with open(clinical_path, 'w') as f:
                json.dump(clinical_data, f, indent=2)
            logger.debug(f"Saved raw clinical data to {clinical_path}")
    
    def _save_metadata(self, patient_id: str, results: Dict[str, Any]) -> None:
        """
        Save preprocessing metadata
        
        Args:
            patient_id: Patient identifier
            results: Pipeline results
        """
        # Convert numpy arrays to lists for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_to_serializable(results)
        
        metadata_path = self.config.get_output_path('final_outputs', patient_id, 'metadata.json')
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        
        with open(metadata_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.debug(f"Saved preprocessing metadata to {metadata_path}")
