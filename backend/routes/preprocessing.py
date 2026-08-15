"""
API routes for ultrasound image preprocessing
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Optional
import uuid
import logging
import os

from preprocessing import UltrasoundPreprocessor
from schemas.preprocessing import (
    PreprocessingResponse,
    PreprocessingConfigResponse
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/preprocessing", tags=["Ultrasound Preprocessing"])

# Initialize preprocessor
preprocessor = UltrasoundPreprocessor()


@router.post("/ultrasound", response_model=PreprocessingResponse)
async def preprocess_ultrasound(
    image: UploadFile = File(...),
    image_id: Optional[str] = Form(None)
):
    """
    Preprocess ultrasound image through the complete pipeline
    
    Pipeline stages:
    1. Resize to target resolution
    2. Speckle noise reduction (Lee filter or SRAD)
    3. CLAHE for contrast enhancement
    4. U-Net segmentation for ovarian follicles
    5. ROI extraction from segmentation
    6. Normalization
    
    Args:
        image: Uploaded ultrasound image file
        image_id: Optional unique identifier (auto-generated if not provided)
        
    Returns:
        Preprocessing results with stage outputs
    """
    try:
        # Generate image ID if not provided
        if not image_id:
            image_id = str(uuid.uuid4())
        
        # Read image data
        image_data = await image.read()
        
        if not image_data:
            raise HTTPException(status_code=400, detail="No image data provided")
        
        # Perform preprocessing
        preprocessing_result = preprocessor.preprocess_image(
            image_data=image_data,
            image_id=image_id,
            filename=image.filename
        )
        
        # Add timestamp
        from datetime import datetime, timezone
        preprocessing_result['preprocessing_timestamp'] = datetime.now(timezone.utc).isoformat()
        
        # Add final image path if available
        if preprocessing_result.get('final_image') is not None:
            from preprocessing.ultrasound_preprocessor import PreprocessingStage
            preprocessing_result['final_image_path'] = preprocessor._get_stage_output_path(
                image_id, 
                PreprocessingStage.NORMALIZED
            )
        
        return preprocessing_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in ultrasound preprocessing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {str(e)}")


@router.get("/config")
async def get_preprocessing_config():
    """
    Get current preprocessing configuration
    
    Returns:
        Dictionary containing all configurable preprocessing parameters
    """
    from preprocessing_config import PreprocessingConfig
    
    return PreprocessingConfigResponse(
        resize_params=PreprocessingConfig.get_resize_params(),
        noise_reduction_params=PreprocessingConfig.get_noise_reduction_params(),
        clahe_params=PreprocessingConfig.get_clahe_params(),
        unet_params=PreprocessingConfig.get_unet_params(),
        normalization_params=PreprocessingConfig.get_normalization_params()
    )


@router.get("/outputs/{image_id}/{stage}")
async def get_stage_output(image_id: str, stage: str):
    """
    Get the output of a specific preprocessing stage
    
    Args:
        image_id: Image identifier
        stage: Preprocessing stage name (original, resized, denoised, clahe_enhanced, segmentation, roi_extracted, normalized)
        
    Returns:
        Image file for the requested stage
    """
    try:
        from preprocessing_config import PreprocessingConfig
        
        # Validate stage name
        valid_stages = list(PreprocessingConfig.STAGE_DIRECTORIES.keys()) + list(PreprocessingConfig.STAGE_DIRECTORIES.values())
        if stage not in valid_stages:
            raise HTTPException(status_code=400, detail=f"Invalid stage: {stage}")
        
        # Get output path
        output_path = PreprocessingConfig.get_output_path(stage, image_id)
        
        if not os.path.exists(output_path):
            raise HTTPException(status_code=404, detail=f"Output not found for stage {stage}")
        
        # Read and return image
        with open(output_path, 'rb') as f:
            image_data = f.read()
        
        from fastapi.responses import Response
        return Response(content=image_data, media_type="image/png")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving stage output: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve output: {str(e)}")


@router.delete("/outputs/{image_id}")
async def delete_preprocessing_outputs(image_id: str):
    """
    Delete all preprocessing outputs for a specific image
    
    Args:
        image_id: Image identifier
        
    Returns:
        Deletion status
    """
    try:
        from preprocessing_config import PreprocessingConfig
        
        deleted_files = []
        errors = []
        
        # Delete outputs from all stages
        for stage_dir in PreprocessingConfig.STAGE_DIRECTORIES.values():
            output_path = PreprocessingConfig.get_output_path(stage_dir, image_id)
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                    deleted_files.append(output_path)
                except Exception as e:
                    errors.append(f"Failed to delete {output_path}: {str(e)}")
        
        return {
            "image_id": image_id,
            "deleted_files": deleted_files,
            "errors": errors,
            "status": "completed" if not errors else "partial_failure"
        }
        
    except Exception as e:
        logger.error(f"Error deleting preprocessing outputs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")
