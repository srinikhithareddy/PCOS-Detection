"""
Pydantic schemas for clinical preprocessing requests and responses
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime


class ClinicalPreprocessingRequest(BaseModel):
    """Request schema for clinical preprocessing"""
    patient_id: str = Field(..., description="Unique identifier for the patient")
    clinical_data: Dict[str, Any] = Field(..., description="Clinical features dictionary")
    mode: str = Field(default="inference", description="Mode: 'train' or 'inference'")
    target: Optional[float] = Field(None, description="Target variable (for training mode)")


class ClinicalPreprocessingResponse(BaseModel):
    """Response schema for clinical preprocessing"""
    patient_id: str
    mode: str
    original_features: List[str]
    selected_features: List[str]
    final_shape: List[int]
    preprocessing_status: str
    error: Optional[str] = None
    preprocessing_timestamp: str
    feature_vector: Optional[List[float]] = None


class ClinicalPreprocessingConfigResponse(BaseModel):
    """Response schema for clinical preprocessing configuration"""
    imputation_params: Dict[str, Any]
    outlier_params: Dict[str, Any]
    encoding_params: Dict[str, Any]
    feature_selection_params: Dict[str, Any]
    normalization_params: Dict[str, Any]
