"""
Facial image processing routes for the API.
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks, Request
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import base64
import time
import uuid

from app.core.image_processor import process_image, async_process_image
from app.db.postgres import PostgresClient, PerceptualHashCache
from app.utils.logging import log_request, log_response, log_job_status
from app.monitoring.prometheus import track_job_status

router = APIRouter(prefix="/api/v1", tags=["facial-processing"])

# Define the request model
class ImageProcessingRequest(BaseModel):
    image: str  # Base64 encoded image
    segmentation_map: Optional[str] = None  # Base64 encoded segmentation map
    job_id: Optional[str] = None  # Optional job ID for tracking
    options: Optional[Dict[str, Any]] = {}  # Processing options

# Define the response models
class ProcessingResponse(BaseModel):
    job_id: str
    status: str
    message: str

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: float
    result: Optional[Dict[str, Any]] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None

# Get PostgreSQL client instance
db_client = PostgresClient()
perceptual_hash_cache = PerceptualHashCache(db_client)

@router.post("/process-image", response_model=ProcessingResponse)
async def process_image_endpoint(
    request_data: ImageProcessingRequest,
    background_tasks: BackgroundTasks,
    request: Request
):
    """
    Process an image to extract facial regions and generate SVG masks.
    
    This endpoint accepts a base64 encoded image and optional segmentation map.
    Processing happens asynchronously, and the client receives a job ID to track progress.
    """
    # Log the incoming request
    log_request(request, request_data)
    
    # Generate a job ID if one is not provided
    job_id = request_data.job_id or str(uuid.uuid4())
    
    # Check if we have a cached result using perceptual hash
    cache_result = await perceptual_hash_cache.get_cached_result(
        request_data.image, 
        request_data.options
    )
    
    if cache_result:
        # Return the cached result
        response = {
            "job_id": job_id,
            "status": "completed",
            "message": "Result retrieved from cache"
        }
        log_response(request, response)
        
        # Update metrics
        track_job_status(job_id, "completed", cache_hit=True)
        
        return response
    
    # Store initial job status
    await db_client.store_job_status(
        job_id=job_id,
        status="queued",
        progress=0.0,
        start_time=time.time()
    )
    
    # Update metrics
    track_job_status(job_id, "queued")
    
    # Add task to background processing
    background_tasks.add_task(
        process_image_task,
        job_id=job_id,
        image_data=request_data.image,
        segmentation_map=request_data.segmentation_map,
        options=request_data.options
    )
    
    # Return response with job ID
    response = {
        "job_id": job_id,
        "status": "queued",
        "message": "Image processing job submitted successfully"
    }
    
    log_response(request, response)
    return response

@router.get("/job-status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str, request: Request):
    """
    Get the status of a processing job.
    
    This endpoint retrieves the current status, progress, and result (if available)
    for a specified job ID.
    """
    log_request(request, {"job_id": job_id})
    
    # Get job status from the database
    job_status = await db_client.get_job_status(job_id)
    
    if not job_status:
        raise HTTPException(status_code=404, detail=f"Job with ID {job_id} not found")
    
    response = JobStatusResponse(
        job_id=job_id,
        status=job_status["status"],
        progress=job_status["progress"],
        result=job_status.get("result"),
        start_time=job_status.get("start_time"),
        end_time=job_status.get("end_time")
    )
    
    log_response(request, response)
    return response

async def process_image_task(job_id: str, image_data: str, segmentation_map: Optional[str], options: Dict[str, Any]):
    """
    Background task for processing an image.
    
    This function is called by the background tasks worker and updates the job status
    in the database as processing progresses.
    """
    try:
        # Update job status to "processing"
        await db_client.update_job_status(
            job_id=job_id,
            status="processing",
            progress=0.1
        )
        
        # Update metrics
        track_job_status(job_id, "processing")
        
        # Log job status update
        log_job_status(job_id, "processing", 0.1)
        
        # Process the image
        result = await async_process_image(
            image_data, 
            segmentation_map, 
            job_id=job_id,
            options=options,
            progress_callback=lambda progress: 
                db_client.update_job_status(job_id, "processing", progress)
        )
        
        # Store the result in the cache
        await perceptual_hash_cache.store_result(
            image_data, 
            result, 
            options
        )
        
        # Update job status to "completed"
        await db_client.update_job_status(
            job_id=job_id,
            status="completed",
            progress=1.0,
            result=result,
            end_time=time.time()
        )
        
        # Update metrics
        track_job_status(job_id, "completed")
        
        # Log job completion
        log_job_status(job_id, "completed", 1.0)
        
    except Exception as e:
        # Update job status to "failed"
        await db_client.update_job_status(
            job_id=job_id,
            status="failed",
            progress=0.0,
            result={"error": str(e)},
            end_time=time.time()
        )
        
        # Update metrics
        track_job_status(job_id, "failed")
        
        # Log error
        log_job_status(job_id, "failed", 0.0, error=str(e))
