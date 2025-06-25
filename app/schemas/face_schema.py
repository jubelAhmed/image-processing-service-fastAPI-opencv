"""
Schema definitions for facial processing API.
"""
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any

class LandmarkPoint(BaseModel):
    x: float
    y: float

class ImageProcessingRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded image")
    landmarks: List[LandmarkPoint] = Field(..., description="Facial landmark points")
    segmentation_map: str = Field(..., description="Base64 encoded segmentation map")


class ProcessingResponse(BaseModel):
    job_id: str | None
    status: str | None
    svg: str | None = None
    mask_contours: Any | None = None