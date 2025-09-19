"""
Service layer for facial processing operations.
"""

from typing import Dict, Any, Optional, List
import json
import hashlib
from datetime import datetime, timedelta
from sqlalchemy import select, update, delete
from sqlalchemy.orm import selectinload
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.database import SessionDep
from src.facial.models import Cache, Job, ProcessingMetrics
from src.facial.schemas import LandmarkPoint
from src.core.utils import logger
from src.facial.exceptions import DatabaseException


class DatabaseService:
    """Database service layer with modern session dependency pattern."""
    
    def __init__(self, session: SessionDep):
        """Initialize database service with session dependency."""
        self.session = session
    
    # ========== JOB MANAGEMENT METHODS ==========
    
    async def store_job_status(
        self, 
        job_id: str, 
        status: str, 
        cache_id: Optional[int] = None,
        error_message: Optional[str] = None
    ) -> None:
        """Store or update job status."""
        try:
            # Check if job exists
            result = await self.session.execute(
                select(Job).where(Job.id == job_id)
            )
            job = result.scalar_one_or_none()
            
            if job:
                # Update existing job
                await self.session.execute(
                    update(Job)
                    .where(Job.id == job_id)
                    .values(
                        status=status,
                        cache_id=cache_id,
                        error_message=error_message,
                        updated_at=datetime.utcnow()
                    )
                )
            else:
                # Create new job
                job = Job(
                    id=job_id,
                    status=status,
                    cache_id=cache_id,
                    error_message=error_message
                )
                self.session.add(job)
            
            await self.session.commit()
            logger.info(f"Job {job_id} status updated to {status}")
            
        except SQLAlchemyError as e:
            await self.session.rollback()
            logger.error(f"Database error storing job status: {e}")
            raise DatabaseException(f"Failed to store job status: {str(e)}")
    
    async def get_job_with_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job with result data from cache."""
        try:
            result = await self.session.execute(
                select(Job)
                .options(selectinload(Job.cache_entry))
                .where(Job.id == job_id)
            )
            job = result.scalar_one_or_none()
            
            if not job:
                return None
            
            job_data = job.to_dict()
            
            # Add result data if available
            if job.cache_entry:
                job_data["result"] = job.cache_entry.result
            
            return job_data
            
        except SQLAlchemyError as e:
            logger.error(f"Database error getting job: {e}")
            raise DatabaseException(f"Failed to get job: {str(e)}")
    
    async def get_job_status(self, job_id: str) -> Optional[str]:
        """Get job status."""
        try:
            result = await self.session.execute(
                select(Job.status).where(Job.id == job_id)
            )
            return result.scalar_one_or_none()
            
        except SQLAlchemyError as e:
            logger.error(f"Database error getting job status: {e}")
            raise DatabaseException(f"Failed to get job status: {str(e)}")
    
    # ========== CACHE MANAGEMENT METHODS ==========
    
    async def store_cache_result(
        self, 
        input_hash: str, 
        result: Dict[str, Any]
    ) -> int:
        """Store processing result in cache."""
        try:
            # Check if cache entry already exists
            result_query = await self.session.execute(
                select(Cache).where(Cache.input_hash == input_hash)
            )
            cache_entry = result_query.scalar_one_or_none()
            
            if cache_entry:
                # Update existing cache entry
                await self.session.execute(
                    update(Cache)
                    .where(Cache.input_hash == input_hash)
                    .values(result=result)
                )
                cache_id = cache_entry.id
            else:
                # Create new cache entry
                cache_entry = Cache(
                    input_hash=input_hash,
                    result=result
                )
                self.session.add(cache_entry)
                await self.session.flush()  # Get the ID
                cache_id = cache_entry.id
            
            await self.session.commit()
            logger.info(f"Cache result stored with ID {cache_id}")
            return cache_id
            
        except SQLAlchemyError as e:
            await self.session.rollback()
            logger.error(f"Database error storing cache result: {e}")
            raise DatabaseException(f"Failed to store cache result: {str(e)}")
    
    async def get_cache_result(self, input_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached processing result."""
        try:
            result = await self.session.execute(
                select(Cache).where(Cache.input_hash == input_hash)
            )
            cache_entry = result.scalar_one_or_none()
            
            if cache_entry:
                return cache_entry.result
            
            return None
            
        except SQLAlchemyError as e:
            logger.error(f"Database error getting cache result: {e}")
            raise DatabaseException(f"Failed to get cache result: {str(e)}")
    
    # ========== METRICS MANAGEMENT METHODS ==========
    
    async def store_processing_metrics(
        self,
        job_id: str,
        processing_time_ms: int,
        image_size_bytes: Optional[int] = None,
        contour_count: Optional[int] = None,
        generator_type: Optional[str] = None
    ) -> None:
        """Store processing metrics."""
        try:
            metrics = ProcessingMetrics(
                job_id=job_id,
                processing_time_ms=processing_time_ms,
                image_size_bytes=image_size_bytes,
                contour_count=contour_count,
                generator_type=generator_type
            )
            
            self.session.add(metrics)
            await self.session.commit()
            logger.info(f"Processing metrics stored for job {job_id}")
            
        except SQLAlchemyError as e:
            await self.session.rollback()
            logger.error(f"Database error storing metrics: {e}")
            raise DatabaseException(f"Failed to store metrics: {str(e)}")
    
    # ========== UTILITY METHODS ==========
    
    def generate_input_hash(
        self, 
        image_data: str, 
        landmarks: List[LandmarkPoint], 
        segmentation_map: str
    ) -> str:
        """Generate hash for input data."""
        # Create a string representation of the input
        input_string = f"{image_data}:{landmarks}:{segmentation_map}"
        return hashlib.sha256(input_string.encode()).hexdigest()
    
    async def cleanup_old_data(self, days: int = 30) -> None:
        """Clean up old cache and job data."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # Delete old cache entries
            await self.session.execute(
                delete(Cache).where(Cache.created_at < cutoff_date)
            )
            
            # Delete old jobs
            await self.session.execute(
                delete(Job).where(Job.created_at < cutoff_date)
            )
            
            # Delete old metrics
            await self.session.execute(
                delete(ProcessingMetrics).where(ProcessingMetrics.created_at < cutoff_date)
            )
            
            await self.session.commit()
            logger.info(f"Cleaned up data older than {days} days")
            
        except SQLAlchemyError as e:
            await self.session.rollback()
            logger.error(f"Database error during cleanup: {e}")
            raise DatabaseException(f"Failed to cleanup old data: {str(e)}")


def get_database_service(session: SessionDep) -> DatabaseService:
    """Get database service with session dependency."""
    return DatabaseService(session)