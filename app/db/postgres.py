"""
Simplified PostgreSQL database client without progress tracking.
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
from app.utils.logging import logger

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
            # Cache table stores the actual results
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS cache (
                    id SERIAL PRIMARY KEY,
                    input_hash VARCHAR(64) UNIQUE,
                    result JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Jobs table only stores status and references cache
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS jobs (
                    id VARCHAR(36) PRIMARY KEY,
                    status VARCHAR(20) NOT NULL CHECK (status IN ('queued', 'processing', 'completed', 'failed')),
                    cache_id INTEGER REFERENCES cache(id),
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for faster lookups
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_cache_input_hash ON cache (input_hash)
            ''')
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs (status)
            ''')
    
    async def store_job_status(self, job_id: str, status: str, cache_id: Optional[int] = None, error_message: Optional[str] = None):
        """Store job status with optional reference to cache."""
        async with self.pool.acquire() as conn:
            await conn.execute('''
                INSERT INTO jobs (id, status, cache_id, error_message, created_at, updated_at)
                VALUES ($1, $2, $3, $4, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT (id) 
                DO UPDATE SET 
                    status = $2,
                    cache_id = $3,
                    error_message = $4,
                    updated_at = CURRENT_TIMESTAMP
            ''', job_id, status, cache_id, error_message)
    
    async def get_job_with_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status and result (if available) by joining with cache table."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow('''
                SELECT 
                    j.id,
                    j.status,
                    j.cache_id,
                    j.error_message,
                    j.created_at,
                    j.updated_at,
                    c.result
                FROM jobs j
                LEFT JOIN cache c ON j.cache_id = c.id
                WHERE j.id = $1
            ''', job_id)
            
            if row:
                job_data = dict(row)
                if job_data["result"]:
                    try:
                        job_data["result"] = json.loads(job_data["result"])
                    except Exception as e:
                        logger.error(f"Error parsing JSON result for job {job_id}: {e}")
                        pass
                return job_data
            return None
    
    async def store_cached_result(self, input_data: Dict[str, Any], result: Dict[str, Any]) -> int:
        """Cache a processing result and return the cache ID."""
        input_hash = self._generate_input_hash(input_data)
        
        async with self.pool.acquire() as conn:
            try:
                row = await conn.fetchrow('''
                    INSERT INTO cache (input_hash, result)
                    VALUES ($1, $2)
                    ON CONFLICT (input_hash) 
                    DO UPDATE SET result = $2
                    RETURNING id
                ''', input_hash, json.dumps(result))
                
                cache_id = row['id']
                logger.debug(f"Successfully stored in cache with ID: {cache_id}")
                return cache_id
            except Exception as e:
                logger.error(f"Error storing in cache: {e}")
                import traceback
                traceback.print_exc()
                raise
    
    async def get_cached_result(self, input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Retrieve a cached result if available."""
        input_hash = self._generate_input_hash(input_data)
        
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow('SELECT id, result FROM cache WHERE input_hash = $1', input_hash)
            if row and row['result']:
                return {
                    'cache_id': row['id'],
                    'result': json.loads(row['result'])
                }
            return None
    
    def _generate_input_hash(self, input_data: Dict[str, Any]) -> str:
        """Generate a hash from input data for cache lookup."""
        serialized = json.dumps(input_data, sort_keys=True)
        return hashlib.sha256(serialized.encode('utf-8')).hexdigest()