"""
API routes for end-to-end preprocessing pipeline
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Optional
import logging

from pipeline import PreprocessingPipeline
from schemas.pipeline import PipelineResponse, PipelineConfigResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/pipeline", tags=["Preprocessing Pipeline"])

# Initialize pipeline
pipeline = PreprocessingPipeline()


@router.post("/preprocess", response_model=PipelineResponse)
async def run_preprocessing_pipeline(
    patient_id: str = Form(...),
    mode: str = Form(default="inference"),
    ultrasound_image: Optional[UploadFile] = File(None),
    clinical_data: Optional[str] = Form(None)
):
    """
    Run the complete end-to-end preprocessing pipeline
    
    This endpoint processes both ultrasound and clinical data through:
    1. Quality Assessment (both modalities)
    2. Ultrasound Preprocessing (resize, denoise, CLAHE, segmentation, ROI extraction)
    3. Clinical Preprocessing (missing values, outliers, encoding, feature selection, normalization)
    
    Args:
        patient_id: Unique identifier for the patient
        mode: 'train' or 'inference'
        ultrasound_image: Optional ultrasound image file
        clinical_data: Optional JSON string of clinical features
        
    Returns:
        Complete preprocessing results with all outputs
    """
    try:
        import json
        
        # Parse clinical data if provided
        clinical_dict = None
        if clinical_data:
            try:
                clinical_dict = json.loads(clinical_data)
            except json.JSONDecodeError as e:
                raise HTTPException(status_code=400, detail=f"Invalid clinical data JSON: {str(e)}")
        
        # Read ultrasound image if provided
        ultrasound_bytes = None
        ultrasound_filename = None
        if ultrasound_image:
            ultrasound_bytes = await ultrasound_image.read()
            ultrasound_filename = ultrasound_image.filename
        
        # Validate at least one modality is provided
        if not ultrasound_bytes and not clinical_dict:
            raise HTTPException(
                status_code=400,
                detail="At least one of ultrasound_image or clinical_data must be provided"
            )
        
        # Run pipeline
        results = pipeline.run_pipeline(
            patient_id=patient_id,
            ultrasound_image=ultrasound_bytes,
            ultrasound_filename=ultrasound_filename,
            clinical_data=clinical_dict,
            mode=mode
        )
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in preprocessing pipeline: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {str(e)}")


@router.get("/config")
async def get_pipeline_config():
    """
    Get current pipeline configuration
    
    Returns:
        Dictionary containing all pipeline configuration parameters
    """
    from pipeline_config import PipelineConfig
    
    return PipelineConfigResponse(pipeline_config=PipelineConfig.get_pipeline_config())


@router.post("/reset")
async def reset_pipeline():
    """
    Reset the pipeline (clear fitted objects and state)
    
    Returns:
        Reset status
    """
    try:
        # Create new pipeline instance
        global pipeline
        pipeline = PreprocessingPipeline()
        
        return {
            "status": "success",
            "message": "Pipeline reset successfully"
        }
        
    except Exception as e:
        logger.error(f"Error resetting pipeline: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")
