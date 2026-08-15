"""
Pydantic schemas for end-to-end preprocessing pipeline requests and responses
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime


class PipelineRequest(BaseModel):
    """Request schema for end-to-end preprocessing pipeline"""
    patient_id: str = Field(..., description="Unique identifier for the patient")
    mode: str = Field(default="inference", description="Mode: 'train' or 'inference'")
    ultrasound_image: Optional[bytes] = Field(None, description="Raw ultrasound image bytes")
    ultrasound_filename: Optional[str] = Field(None, description="Original ultrasound filename")
    clinical_data: Optional[Dict[str, Any]] = Field(None, description="Clinical features dictionary")


class PipelineResponse(BaseModel):
    """Response schema for preprocessing pipeline results"""
    patient_id: str
    mode: str
    pipeline_start_time: str
    pipeline_end_time: Optional[str]
    pipeline_duration_seconds: Optional[float]
    ultrasound_available: bool
    clinical_available: bool
    ultrasound_pipeline: Dict[str, Any]
    clinical_pipeline: Dict[str, Any]
    final_outputs: Dict[str, Any]
    pipeline_status: str
    errors: List[str]
    validation: Optional[Dict[str, Any]] = None


class PipelineConfigResponse(BaseModel):
    """Response schema for pipeline configuration"""
    pipeline_config: Dict[str, Any]
