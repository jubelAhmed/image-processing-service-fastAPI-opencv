"""
Schema definitions for facial processing API.
"""
from pydantic import BaseModel, Field
from typing import Dict, List, Optional

class LandmarkPoint(BaseModel):
    x: float
    y: float

class ImageProcessingRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded image")
    landmarks: List[LandmarkPoint] = Field(..., description="Facial landmark points")
    segmentation_map: str = Field(..., description="Base64 encoded segmentation map")

class JobStatusResponse(BaseModel):
    id: str
    status: str

class ProcessingResponse(BaseModel):
    svg: Optional[str] = Field(None)
    mask_contours: Optional[Dict[str, List[List[float]]]] = Field(None)
    job_id: Optional[str] = Field(None)
    status: Optional[str] = Field(None)