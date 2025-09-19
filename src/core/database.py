"""
Database session management and configuration for SQLAlchemy.
"""

from typing import AsyncGenerator, Annotated
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from fastapi import Depends
from src.core.config import config
from src.core.exceptions import InternalServerException
from src.core.utils import logger


class DatabaseManager:
    """Manages database connections and sessions using SQLAlchemy."""
    
    def __init__(self, database_url: str = None):
        """
        Initialize database manager.
        
        Args:
            database_url: Database connection URL (defaults to config value)
        """
        self.database_url = database_url or self._get_database_url()
        self.engine = None
        self.session_factory = None
    
    def _get_database_url(self) -> str:
        """Get database URL from configuration."""
        if not config.db.use_database:
            return "sqlite:///./facial_api.db"  # Default to SQLite when disabled
        
        # Simple conversion to async format
        db_url = config.db.connection_string
        if db_url.startswith("postgresql://"):
            return db_url.replace("postgresql://", "postgresql+asyncpg://", 1)
        elif db_url.startswith("postgres://"):
            return db_url.replace("postgres://", "postgresql+asyncpg://", 1)
        
        return db_url
    
    async def initialize(self) -> None:
        """Initialize database engine and session factory."""
        try:
            self.engine = create_async_engine(
                self.database_url,
                echo=config.debug,  # Log SQL queries in debug mode
                pool_pre_ping=True,  # Verify connections before use
                pool_recycle=3600,   # Recycle connections every hour
            )
            
            self.session_factory = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            logger.info("Database engine and session factory initialized")
            
        except Exception as e:
            raise InternalServerException(f"Failed to initialize database: {str(e)}")
    
    async def close(self) -> None:
        """Close database engine and all connections."""
        if self.engine:
            await self.engine.dispose()
            logger.info("Database engine closed")
    
    async def create_tables(self) -> None:
        """Create all database tables."""
        if not self.engine:
            raise InternalServerException("Database engine not initialized")
        
        try:
            from src.core.models import Base
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created successfully")
            
        except Exception as e:
            raise InternalServerException(f"Failed to create tables: {str(e)}")


# Global database manager instance
db_manager = DatabaseManager()


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency to get database session.
    
    Yields:
        AsyncSession: Database session
    """
    if not db_manager.session_factory:
        raise InternalServerException("Database not initialized")
    
    async with db_manager.session_factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# Type alias for dependency injection
SessionDep = Annotated[AsyncSession, Depends(get_session)]


async def create_db_and_tables():
    """Create database tables on startup."""
    try:
        await db_manager.initialize()
        await db_manager.create_tables()
        logger.info("Database initialization completed")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise
