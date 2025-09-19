"""
Pydantic schemas for facial processing API.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Tuple

# Type alias for mask contours
MaskContours = Dict[int, List[Tuple[int, int]]]


class LandmarkPoint(BaseModel):
    """Facial landmark point."""
    x: float
    y: float


class ImageProcessingRequest(BaseModel):
    """Request schema for image processing."""
    image: str = Field(..., description="Base64 encoded image")
    landmarks: List[LandmarkPoint] = Field(..., description="Facial landmark points")
    segmentation_map: str = Field(..., description="Base64 encoded segmentation map")


class JobStatusResponse(BaseModel):
    """Response schema for job status."""
    job_id: str
    status: str
    message: Optional[str] = None


class ProcessingResponse(BaseModel):
    """Response schema for completed processing."""
    svg: Optional[str] = None
    mask_contours: Optional[MaskContours] = None
    status: str = "completed"


class ProcessingErrorResponse(BaseModel):
    """Response schema for processing errors."""
    detail: str
    error_code: str
    job_id: Optional[str] = None


class RateLimitResponse(BaseModel):
    """Response schema for rate limit exceeded."""
    detail: str
    error_code: str = "RATE_LIMIT_EXCEEDED"
    rate_limit: Dict[str, Any]
