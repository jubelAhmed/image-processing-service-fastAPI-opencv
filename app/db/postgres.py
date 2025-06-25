"""
PostgreSQL database client and caching functionality.
"""

from typing import Dict, Any, Optional, List
import json
import hashlib
import time
import asyncpg
import cv2
import numpy as np
import base64
from app.schemas.face_schema import LandmarkPoint

class PostgresClient:
    """Client for interacting with PostgreSQL database."""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.pool = None
    
    async def connect(self):
        """Connect to PostgreSQL and initialize the connection pool."""
        self.pool = await asyncpg.create_pool(self.connection_string)
    
    async def disconnect(self):
        """Close all connections in the pool."""
        if self.pool:
            await self.pool.close()
    
    async def create_tables(self):
        """Create necessary tables if they don't exist."""
        async with self.pool.acquire() as conn:
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS cache (
                    id SERIAL PRIMARY KEY,
                    input_hash VARCHAR(64) UNIQUE,
                    result JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS jobs (
                    id VARCHAR(36) PRIMARY KEY,
                    status VARCHAR(20) NOT NULL,
                    result JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create index for faster lookups
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_cache_input_hash ON cache (input_hash)
            ''')
    
    async def store_job(self, job_id: str, status: str, result: Optional[Dict[str, Any]] = None):
        """Store or update a job in the database."""
        async with self.pool.acquire() as conn:
            await conn.execute('''
                INSERT INTO jobs (id, status, result, created_at, updated_at)
                VALUES ($1, $2, $3, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT (id) 
                DO UPDATE SET 
                    status = $2,
                    result = $3,
                    updated_at = CURRENT_TIMESTAMP
            ''', job_id, status, json.dumps(result) if result else None)
    
    async def store_job_status(self, job_id: str, status: str, progress: float, start_time: Optional[float] = None, end_time: Optional[float] = None, result: Optional[Dict[str, Any]] = None):
        """Store job status with progress information."""
        job_data = {
            "status": status,
            "progress": progress
        }
        
        if start_time:
            job_data["start_time"] = start_time
            
        if end_time:
            job_data["end_time"] = end_time
            
        if result:
            job_data["result"] = result
            
        async with self.pool.acquire() as conn:
            await conn.execute('''
                INSERT INTO jobs (id, status, result, created_at, updated_at)
                VALUES ($1, $2, $3, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT (id) 
                DO UPDATE SET 
                    status = $2,
                    result = $3,
                    updated_at = CURRENT_TIMESTAMP
            ''', job_id, status, json.dumps(job_data))
            
    async def update_job_status(self, job_id: str, status: str, progress: float, result: Optional[Dict[str, Any]] = None, end_time: Optional[float] = None):
        """Update an existing job's status and progress."""
        # Get the current job data
        job_data = await self.get_job_status(job_id) or {}
        
        # Update the status and progress
        job_data["status"] = status
        job_data["progress"] = progress
        
        if result:
            job_data["result"] = result
            
        if end_time:
            job_data["end_time"] = end_time
            
        # Store the updated job
        await self.store_job_status(
            job_id=job_id,
            status=status,
            progress=progress,
            start_time=job_data.get("start_time"),
            end_time=end_time,
            result=result
        )
    
    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status with progress information."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow('SELECT * FROM jobs WHERE id = $1', job_id)
            if row:
                job_data = dict(row)
                if job_data["result"]:
                    try:
                        job_data["result"] = json.loads(job_data["result"])
                    except:
                        pass
                return job_data
            return None
    
    async def store_cached_result(self, input_data: Dict[str, Any], result: Dict[str, Any]):
        """Cache a processing result."""
        input_hash = self._generate_input_hash(input_data)
        
        async with self.pool.acquire() as conn:
            await conn.execute('''
                INSERT INTO cache (input_hash, result)
                VALUES ($1, $2)
                ON CONFLICT (input_hash) 
                DO UPDATE SET result = $2
            ''', input_hash, json.dumps(result))
    
    def get_cached_result(self, input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Retrieve a cached result if available."""
        input_hash = self._generate_input_hash(input_data)
        
        with self.pool.acquire() as conn:
            row = conn.fetchrow('SELECT result FROM cache WHERE input_hash = $1', input_hash)
            if row and row['result']:
                return json.loads(row['result'])
            return None
    
    def _generate_input_hash(self, input_data: Dict[str, Any]) -> str:
        """Generate a hash from input data for cache lookup."""
        # For images, we can use perceptual hashing to identify similar images
        # For simplicity, we'll use a regular hash of the serialized input
        serialized = json.dumps(input_data, sort_keys=True)
        return hashlib.sha256(serialized.encode('utf-8')).hexdigest()


