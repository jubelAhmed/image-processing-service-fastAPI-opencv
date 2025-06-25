"""
Facial image processing routes for the API.
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks, Request, Depends
from typing import Dict, Any, Optional, Union
import time
import uuid

from app.db.postgres import PostgresClient
from app.db.perceptual_caching import PerceptualHashCache
from app.utils.logging import log_request, log_response, log_job_status
from app.monitoring.prometheus import track_job_status
from app.schemas.face_schema import (
    LandmarkPoint, 
    ImageProcessingRequest,
    ProcessingResponse, JobStatusResponse)

from app.utils.logging import logger

router = APIRouter(prefix="/api/v1", tags=["facial-processing"])

# Dependency to get PostgreSQL client
async def get_postgres_client():
    from app.main import app
    from app.config import config
    
    # Return None if database is disabled
    if not config.db.use_database or not hasattr(app.state, "postgres_client"):
        return None
    return app.state.postgres_client

# Dependency to get cache client
async def get_perceptual_hash_cache(postgres_client: Optional[PostgresClient] = Depends(get_postgres_client)):
    if postgres_client is None:
        return None
    return PerceptualHashCache(postgres_client)


@router.post("/frontal/crop/submit", response_model=Union[ProcessingResponse, JobStatusResponse])
async def process_image_endpoint(
    request_data: ImageProcessingRequest,
    background_tasks: BackgroundTasks,
    request: Request,
    postgres_client: Optional[PostgresClient] = Depends(get_postgres_client),
    perceptual_hash_cache: Optional[PerceptualHashCache] = Depends(get_perceptual_hash_cache)
):
    """Process an image to extract facial regions and generate SVG masks."""
    log_request(request, request_data)
    
    # Check cache first
    if perceptual_hash_cache:
        cache_result = await perceptual_hash_cache.get_cached_result(
            request_data.image,
            request_data.landmarks,
            request_data.segmentation_map
        )
        
        if cache_result:    
            # Extract the actual result data from cache
            cached_data = cache_result.get('result', {})
            try:
                svg_data, mask_contours = extract_result_data(cached_data)
                response = {
                    "svg": svg_data,
                    "mask_contours": mask_contours,
                }
            except Exception as e:
                logger.error(f"Error extracting cached result data: {e}")
                response = {
                    "status": "error",
                    "job_id": ""
                }
            
            log_response(request, response)
            return response
    
    job_id = str(uuid.uuid4())
    # No cache hit - queue for processing
    if postgres_client:
        await postgres_client.store_job_status(
            job_id=job_id,
            status="queued"
        )
    
    track_job_status(job_id, "queued")
    
    background_tasks.add_task(
        process_image_task,
        job_id=job_id,
        image_data=request_data.image,
        segmentation_map=request_data.segmentation_map,
        landmarks=request_data.landmarks,
        postgres_client=postgres_client,
        perceptual_hash_cache=perceptual_hash_cache
    )
    
    response = {"job_id": job_id, "status": "queued"}
    log_response(request, response)
    return response


@router.get("/frontal/crop/status/{job_id}")
async def get_job_status(
    job_id: str, 
    request: Request,
    postgres_client: Optional[PostgresClient] = Depends(get_postgres_client)
):
    """
    Get the status of a processing job.
    
    Returns:
    - If completed: {"svg": "base64str", "mask_contours": {"1": [...]}}
    - If pending/processing: {"status": "pending"} or {"status": "processing"}
    - If failed: {"status": "failed", "error": "error message"}
    """
    log_request(request, {"job_id": job_id})
    
    if not postgres_client:
        raise HTTPException(status_code=503, detail="Database functionality is disabled")
    
    # Get job status with result from joined query
    job_data = await postgres_client.get_job_with_result(job_id)
    
    if not job_data:
        raise HTTPException(status_code=404, detail=f"Job with ID {job_id} not found")
    
    
    status = job_data.get("status", "unknown")
    
    if status == "completed":
        # Extract and validate the result data
        result_data = job_data.get("result")
        
        try:
            svg_data, mask_contours = extract_result_data(result_data)
            response = {
                "svg": svg_data,
                "mask_contours": mask_contours
            }
        except Exception as e:
            logger.error(f"Error extracting result data for job {job_id}: {e}")
            # Fallback response
            response = {
                "status": "completed",
                "error": f"Result data extraction failed: {str(e)}",
            }
            
    elif status == "failed":
        response = {
            "status": "failed",
            "error": job_data.get("error_message", "Processing failed")
        }
        
    elif status == "processing":
        response = {
            "status": "processing",
        }
        
    else:
        # queued or other statuses
        response = {
            "status": status
        }
    
    log_response(request, response)
    return response

def extract_result_data(result_data):
    """Extract SVG and mask contours from the result data."""
    with open("test.txt", "w") as f:
        f.write("Extracting result data...\n")
        f.write(f"Result data: {result_data}\n")
   
    svg_data = result_data.get("svg", "")
    mask_contours = result_data.get("mask_contours", {})
    
    if not svg_data:
        raise ValueError("SVG data is missing in the result")
    
    if not isinstance(mask_contours, dict):
        raise ValueError("Mask contours must be a dictionary")
    
    return svg_data, mask_contours

async def process_image_task(
    job_id: str, 
    image_data: str,
    segmentation_map: str,
    landmarks: list[LandmarkPoint],
    postgres_client: Optional[PostgresClient],
    perceptual_hash_cache: Optional[PerceptualHashCache]
):
    """Background task for processing an image."""
    from app.core.facial_segmentation_processor import FacialSegmentationProcessor
    facial_processor = FacialSegmentationProcessor()
    
    try:
        # Update to processing
        if postgres_client:
            await postgres_client.store_job_status(
                job_id=job_id,
                status="processing"            )
        
        track_job_status(job_id, "processing")
        log_job_status(job_id, "processing", 0.1)
    

        
        svg_base64, contours = await facial_processor.process_image(
            image_data, 
            segmentation_map, 
            landmarks,
        )

        result = {
            "svg": svg_base64,
            "mask_contours": contours
        }
        
        # Store result in cache and get cache ID
        cache_id = None
        if perceptual_hash_cache:
            cache_id = await perceptual_hash_cache.store_result(
                image_data, 
                result,
                landmarks,
                segmentation_map
            )
        
        # Update job status with reference to cache
        if postgres_client:
            await postgres_client.store_job_status(
                job_id=job_id,
                status="completed",
                cache_id=cache_id  # Reference the cache entry
            )
        
        track_job_status(job_id, "completed")
        log_job_status(job_id, "completed", 1.0)
        
    except Exception as e:
        if postgres_client:
            await postgres_client.store_job_status(
                job_id=job_id,
                status="failed",
                error_message=str(e)
            )
        
        track_job_status(job_id, "failed")
        log_job_status(job_id, "failed", 0.0, error=str(e))