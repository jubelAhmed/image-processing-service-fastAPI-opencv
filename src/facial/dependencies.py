"""
Dependencies for facial processing endpoints.
"""

from typing import Optional
from fastapi import Depends, HTTPException, status
from src.core.database import SessionDep
from src.facial.service import DatabaseService, get_database_service
from src.facial.perceptual_caching import PerceptualHashCache
from src.core.config import config


async def get_db_service(session: SessionDep) -> Optional[DatabaseService]:
    """Get database service with session dependency."""
    # Return None if database is disabled
    if not config.db.use_database:
        return None
    return get_database_service(session)


async def get_perceptual_hash_cache(
    db_service: Optional[DatabaseService] = Depends(get_db_service)
) -> Optional[PerceptualHashCache]:
    """Get perceptual hash cache with database service dependency."""
    if db_service is None:
        return None
    return PerceptualHashCache(db_service)
