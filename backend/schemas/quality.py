"""
Pydantic schemas for quality assessment requests and responses
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime


class ImageQualityRequest(BaseModel):
    """Request schema for image quality assessment"""
    image_id: str = Field(..., description="Unique identifier for the image")
    filename: str = Field(..., description="Original filename")
    image_data: bytes = Field(..., description="Raw image bytes")


class ClinicalQualityRequest(BaseModel):
    """Request schema for clinical data quality assessment"""
    patient_id: str = Field(..., description="Unique identifier for the patient")
    clinical_data: Dict[str, Any] = Field(..., description="Clinical features dictionary")


class QualityMetrics(BaseModel):
    """Image quality metrics"""
    quality_score: Optional[float] = Field(None, description="Overall quality score (0-100, higher is better)")
    sharpness: Optional[float] = Field(None, description="Sharpness score (Laplacian variance)")
    mean_brightness: Optional[float] = Field(None, description="Mean brightness")
    std_brightness: Optional[float] = Field(None, description="Standard deviation of brightness")
    contrast: Optional[float] = Field(None, description="Image contrast")


class ImageDimensions(BaseModel):
    """Image dimensions"""
    height: int = Field(..., description="Image height in pixels")
    width: int = Field(..., description="Image width in pixels")


class ImageQualityResponse(BaseModel):
    """Response schema for image quality assessment"""
    image_id: str
    filename: str
    assessment_timestamp: str
    image_dimensions: Optional[ImageDimensions] = None
    quality_metrics: QualityMetrics
    quality_score: Optional[float] = None
    quality_category: str = Field(..., description="good, poor, or unusable")
    processing_decision: str = Field(..., description="continue, enhance, or flag")
    requires_enhancement: bool
    flagged_for_review: bool
    assessment_status: str
    failure_reason: Optional[str] = None


class MissingValueAnalysis(BaseModel):
    """Missing value analysis results"""
    missing_features: List[str]
    missing_count: int
    missing_percent: float
    critical_missing_features: List[str]
    critical_missing_count: int
    exceeds_threshold: bool
    has_critical_missing: bool


class RangeViolation(BaseModel):
    """Individual range violation"""
    feature: str
    value: float
    expected_min: Optional[float]
    expected_max: Optional[float]
    violation_type: str


class RangeValidation(BaseModel):
    """Range validation results"""
    range_violations: List[RangeViolation]
    violation_count: int
    violated_features: List[str]


class OutlierInfo(BaseModel):
    """Individual outlier information"""
    feature: str
    value: float
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    z_score: Optional[float] = None
    threshold: Optional[float] = None
    method: str


class OutlierDetection(BaseModel):
    """Outlier detection results"""
    outliers: List[OutlierInfo]
    outlier_count: int
    outlier_features: List[str]
    method_used: str


class ConsistencyViolation(BaseModel):
    """Individual consistency violation"""
    check_type: str
    reported_value: float
    calculated_value: float
    difference_percent: float
    tolerance_percent: float


class ConsistencyChecks(BaseModel):
    """Consistency check results"""
    consistency_violations: List[ConsistencyViolation]
    violation_count: int
    violated_checks: List[str]


class ClinicalQualityResponse(BaseModel):
    """Response schema for clinical data quality assessment"""
    patient_id: str
    assessment_timestamp: str
    reliability_score: float = Field(..., ge=0.0, le=1.0)
    quality_category: str = Field(..., description="high, medium, low, or unusable")
    missing_value_analysis: MissingValueAnalysis
    range_validation: RangeValidation
    outlier_detection: OutlierDetection
    consistency_checks: ConsistencyChecks
    assessment_status: str
    error_message: Optional[str] = None
    requires_manual_review: bool


class CombinedQualityRequest(BaseModel):
    """Request schema for combined quality assessment"""
    patient_id: str
    image_id: str
    filename: str
    image_data: bytes
    clinical_data: Dict[str, Any]


class CombinedQualityResponse(BaseModel):
    """Response schema for combined quality assessment"""
    patient_id: str
    image_id: str
    assessment_timestamp: str
    image_quality: ImageQualityResponse
    clinical_quality: ClinicalQualityResponse
    overall_reliability_score: float
    overall_quality_category: str
    can_proceed_to_preprocessing: bool
