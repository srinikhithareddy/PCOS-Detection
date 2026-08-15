"""
Pydantic schemas for preprocessing requests and responses
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime


class PreprocessingRequest(BaseModel):
    """Request schema for ultrasound preprocessing"""
    image_id: str = Field(..., description="Unique identifier for the image")
    filename: str = Field(..., description="Original filename")
    image_data: bytes = Field(..., description="Raw image bytes")


class PreprocessingResponse(BaseModel):
    """Response schema for preprocessing results"""
    image_id: str
    filename: str
    stages_completed: List[str]
    stage_outputs: Dict[str, str]
    final_image_path: Optional[str] = None
    preprocessing_status: str
    error: Optional[str] = None
    preprocessing_timestamp: str


class StageOutput(BaseModel):
    """Individual preprocessing stage output"""
    stage_name: str
    output_path: str
    timestamp: str


class PreprocessingConfigResponse(BaseModel):
    """Response schema for preprocessing configuration"""
    resize_params: Dict[str, Any]
    noise_reduction_params: Dict[str, Any]
    clahe_params: Dict[str, Any]
    unet_params: Dict[str, Any]
    normalization_params: Dict[str, Any]
