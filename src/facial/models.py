"""
Database models for facial processing.
"""

from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, JSON, CheckConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from src.core.models import Base


class Cache(Base):
    """Model for caching processing results."""
    
    __tablename__ = "cache"
    
    id = Column(Integer, primary_key=True, index=True)
    input_hash = Column(String(64), unique=True, index=True, nullable=False)
    result = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationship to jobs that reference this cache entry
    jobs = relationship("Job", back_populates="cache_entry")


class Job(Base):
    """Model for tracking job status and results."""
    
    __tablename__ = "jobs"
    
    id = Column(String(36), primary_key=True, index=True)
    status = Column(
        String(20), 
        CheckConstraint("status IN ('queued', 'processing', 'completed', 'failed')"),
        nullable=False
    )
    cache_id = Column(Integer, ForeignKey("cache.id"), nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationship to cache entry
    cache_entry = relationship("Cache", back_populates="jobs")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert job instance to dictionary."""
        return {
            "id": self.id,
            "status": self.status,
            "cache_id": self.cache_id,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "result": self.cache_entry.result if self.cache_entry else None
        }
    
    def __repr__(self) -> str:
        return f"<Job(id='{self.id}', status='{self.status}', cache_id={self.cache_id})>"


class ProcessingMetrics(Base):
    """Model for storing processing performance metrics."""
    
    __tablename__ = "processing_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String(36), ForeignKey("jobs.id"), nullable=False)
    processing_time_ms = Column(Integer, nullable=False)
    image_size_bytes = Column(Integer, nullable=True)
    contour_count = Column(Integer, nullable=True)
    generator_type = Column(String(20), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationship to job
    job = relationship("Job", foreign_keys=[job_id])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics instance to dictionary."""
        return {
            "id": self.id,
            "job_id": self.job_id,
            "processing_time_ms": self.processing_time_ms,
            "image_size_bytes": self.image_size_bytes,
            "contour_count": self.contour_count,
            "generator_type": self.generator_type,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
    
    def __repr__(self) -> str:
        return f"<ProcessingMetrics(job_id='{self.job_id}', time_ms={self.processing_time_ms})>"
