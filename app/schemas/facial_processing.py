"""
Schema definitions for facial processing API.
"""
from pydantic import BaseModel
from typing import Dict, Any, Optional, List

class Landmark(BaseModel):
    """Facial landmark coordinate."""
    x: float
    y: float

class ImageProcessingRequest(BaseModel):
    """Request model for image processing endpoints."""
    image: str  # Base64 encoded image
    segmentation_map: str  # Base64 encoded segmentation map
    landmarks: List[Landmark]  # Landmarks for facial features
    job_id: Optional[str] = None  # Optional job ID for tracking
    options: Optional[Dict[str, Any]] = {}  # Processing options

class ProcessingResponse(BaseModel):
    """Response model for asynchronous image processing submission."""
    job_id: str
    status: str
    message: str

class JobStatusResponse(BaseModel):
    """Response model for job status retrieval."""
    job_id: str
    status: str
    progress: float
    result: Optional[Dict[str, Any]] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None

class SynchronousProcessingResponse(BaseModel):
    """Response model for synchronous image processing."""
    success: bool
    result: Dict[str, Any]
    processing_time: float
