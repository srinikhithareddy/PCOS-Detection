"""
API routes for clinical data preprocessing
"""

from fastapi import APIRouter, HTTPException
from typing import Optional
import logging

from clinical_preprocessing import ClinicalPreprocessor
from schemas.clinical_preprocessing import (
    ClinicalPreprocessingResponse,
    ClinicalPreprocessingConfigResponse
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/clinical-preprocessing", tags=["Clinical Preprocessing"])

# Initialize preprocessor
preprocessor = ClinicalPreprocessor()


@router.post("/preprocess", response_model=ClinicalPreprocessingResponse)
async def preprocess_clinical_data(request: dict):
    """
    Preprocess clinical data through the complete pipeline
    
    Pipeline stages:
    1. Missing value handling
    2. Outlier handling
    3. Range & consistency validation
    4. Categorical encoding
    5. Feature selection
    6. Normalization
    
    Args:
        request: Dictionary containing patient_id, clinical_data, mode, and optional target
        
    Returns:
        Preprocessing results with feature vector
    """
    try:
        patient_id = request.get("patient_id")
        clinical_data = request.get("clinical_data")
        mode = request.get("mode", "inference")
        target = request.get("target")
        
        if not patient_id:
            raise HTTPException(status_code=400, detail="patient_id is required")
        
        if not clinical_data:
            raise HTTPException(status_code=400, detail="clinical_data is required")
        
        if mode not in ['train', 'inference']:
            raise HTTPException(status_code=400, detail="mode must be 'train' or 'inference'")
        
        # Convert target to numpy array if provided
        if target is not None:
            import numpy as np
            target = np.array([target])
        
        # Perform preprocessing
        if mode == 'train':
            feature_vector, metadata = preprocessor.fit_transform(clinical_data, target)
        else:
            feature_vector, metadata = preprocessor.transform(clinical_data)
        
        # Add timestamp
        from datetime import datetime, timezone
        metadata['preprocessing_timestamp'] = datetime.now(timezone.utc).isoformat()
        
        # Add patient_id and mode to response
        metadata['patient_id'] = patient_id
        metadata['mode'] = mode
        metadata['feature_vector'] = feature_vector.tolist() if isinstance(feature_vector, np.ndarray) else list(feature_vector)
        
        return metadata
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in clinical preprocessing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {str(e)}")


@router.get("/config")
async def get_clinical_preprocessing_config():
    """
    Get current clinical preprocessing configuration
    
    Returns:
        Dictionary containing all configurable preprocessing parameters
    """
    from clinical_preprocessing_config import ClinicalPreprocessingConfig
    
    return ClinicalPreprocessingConfigResponse(
        imputation_params=ClinicalPreprocessingConfig.get_imputation_params(),
        outlier_params=ClinicalPreprocessingConfig.get_outlier_params(),
        encoding_params=ClinicalPreprocessingConfig.get_encoding_params(),
        feature_selection_params=ClinicalPreprocessingConfig.get_feature_selection_params(),
        normalization_params=ClinicalPreprocessingConfig.get_normalization_params()
    )


@router.post("/reset")
async def reset_preprocessor():
    """
    Reset the preprocessor (clear fitted objects)
    
    Returns:
        Reset status
    """
    try:
        # Create new preprocessor instance
        global preprocessor
        preprocessor = ClinicalPreprocessor()
        
        return {
            "status": "success",
            "message": "Preprocessor reset successfully"
        }
        
    except Exception as e:
        logger.error(f"Error resetting preprocessor: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")
