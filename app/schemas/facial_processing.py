"""
Schema definitions for facial processing API.
"""
from pydantic import BaseModel, Field
from typing import Dict, List

class LandmarkPoint(BaseModel):
    x: float
    y: float

class SegmentationRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded image")
    landmarks: List[LandmarkPoint] = Field(..., description="Facial landmark points")
    segmentation_map: str = Field(..., description="Base64 encoded segmentation map")

class JobResponse(BaseModel):
    id: str
    status: str

class SegmentationResponse(BaseModel):
    svg: str
    mask_contours: Dict[str, List[List[float]]]