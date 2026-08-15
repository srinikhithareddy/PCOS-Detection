"""
API routes for quality assessment
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Optional
import uuid
import logging

from quality_assessment import ImageQualityAssessor, ClinicalQualityAssessor
from schemas.quality import (
    ImageQualityResponse,
    ClinicalQualityResponse,
    CombinedQualityResponse
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/quality", tags=["Quality Assessment"])

# Initialize assessors
image_assessor = ImageQualityAssessor()
clinical_assessor = ClinicalQualityAssessor()


@router.post("/image", response_model=ImageQualityResponse)
async def assess_image_quality(
    image: UploadFile = File(...),
    image_id: Optional[str] = Form(None)
):
    """
    Assess ultrasound image quality using BRISQUE
    
    Args:
        image: Uploaded image file
        image_id: Optional unique identifier (auto-generated if not provided)
        
    Returns:
        Image quality assessment report
    """
    try:
        # Generate image ID if not provided
        if not image_id:
            image_id = str(uuid.uuid4())
        
        # Read image data
        image_data = await image.read()
        
        if not image_data:
            raise HTTPException(status_code=400, detail="No image data provided")
        
        # Perform quality assessment
        quality_report = image_assessor.assess_image_quality(
            image_data=image_data,
            image_id=image_id,
            filename=image.filename
        )
        
        return quality_report
        
    except Exception as e:
        logger.error(f"Error in image quality assessment: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Quality assessment failed: {str(e)}")


@router.post("/clinical", response_model=ClinicalQualityResponse)
async def assess_clinical_quality(request: dict):
    """
    Assess clinical data quality
    
    Args:
        request: Dictionary containing patient_id and clinical_data
        
    Returns:
        Clinical quality assessment report
    """
    try:
        patient_id = request.get("patient_id")
        clinical_data = request.get("clinical_data")
        
        if not patient_id:
            raise HTTPException(status_code=400, detail="patient_id is required")
        
        if not clinical_data:
            raise HTTPException(status_code=400, detail="clinical_data is required")
        
        # Perform quality assessment
        quality_report = clinical_assessor.assess_clinical_data_quality(
            clinical_data=clinical_data,
            patient_id=patient_id
        )
        
        return quality_report
        
    except Exception as e:
        logger.error(f"Error in clinical quality assessment: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Quality assessment failed: {str(e)}")


@router.post("/combined", response_model=CombinedQualityResponse)
async def assess_combined_quality(
    image: UploadFile = File(...),
    patient_id: str = Form(...),
    image_id: Optional[str] = Form(None),
    clinical_data: str = Form(...)  # JSON string of clinical data
):
    """
    Assess both image and clinical data quality together
    
    Args:
        image: Uploaded image file
        patient_id: Unique patient identifier
        image_id: Optional unique image identifier
        clinical_data: JSON string containing clinical features
        
    Returns:
        Combined quality assessment report
    """
    try:
        import json
        
        # Generate image ID if not provided
        if not image_id:
            image_id = str(uuid.uuid4())
        
        # Parse clinical data
        try:
            clinical_data_dict = json.loads(clinical_data)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON in clinical_data")
        
        # Read image data
        image_data = await image.read()
        
        if not image_data:
            raise HTTPException(status_code=400, detail="No image data provided")
        
        # Perform image quality assessment
        image_quality_report = image_assessor.assess_image_quality(
            image_data=image_data,
            image_id=image_id,
            filename=image.filename
        )
        
        # Perform clinical quality assessment
        clinical_quality_report = clinical_assessor.assess_clinical_data_quality(
            clinical_data=clinical_data_dict,
            patient_id=patient_id
        )
        
        # Calculate overall reliability
        image_score = 1.0 if image_quality_report['quality_category'] == 'good' else \
                     0.5 if image_quality_report['quality_category'] == 'poor' else 0.0
        
        clinical_score = clinical_quality_report['reliability_score']
        
        # Weighted average (equal weights for now)
        overall_reliability_score = (image_score + clinical_score) / 2.0
        
        # Determine overall quality category
        if overall_reliability_score >= 0.8:
            overall_category = "high"
        elif overall_reliability_score >= 0.6:
            overall_category = "medium"
        elif overall_reliability_score >= 0.4:
            overall_category = "low"
        else:
            overall_category = "unusable"
        
        # Determine if can proceed to preprocessing
        can_proceed = (
            image_quality_report['processing_decision'] != 'flag' and
            clinical_quality_report['quality_category'] != 'unusable'
        )
        
        return {
            "patient_id": patient_id,
            "image_id": image_id,
            "assessment_timestamp": image_quality_report['assessment_timestamp'],
            "image_quality": image_quality_report,
            "clinical_quality": clinical_quality_report,
            "overall_reliability_score": overall_reliability_score,
            "overall_quality_category": overall_category,
            "can_proceed_to_preprocessing": can_proceed
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in combined quality assessment: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Quality assessment failed: {str(e)}")


@router.get("/config/thresholds")
async def get_quality_thresholds():
    """
    Get current quality assessment thresholds
    
    Returns:
        Dictionary containing all configurable thresholds
    """
    from quality_config import QualityConfig
    
    return {
        "image_quality": QualityConfig.get_image_quality_thresholds(),
        "clinical_ranges": QualityConfig.CLINICAL_FEATURE_RANGES,
        "missing_value_threshold": QualityConfig.MAX_MISSING_VALUES_PERCENT,
        "outlier_detection_method": QualityConfig.OUTLIER_DETECTION_METHOD,
        "primary_image_metric": QualityConfig.PRIMARY_IMAGE_QUALITY_METRIC
    }
