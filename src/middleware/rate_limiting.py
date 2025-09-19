"""
Rate limiting configuration using slowapi library.
"""

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request, HTTPException, status
from src.core.config import config
from src.core.utils import logger

# Initialize limiter with Redis backend (fallback to memory)
try:
    # Try Redis first
    limiter = Limiter(
        key_func=get_remote_address,
        storage_uri="redis://localhost:6379",
        default_limits=[f"{config.rate_limit.requests_per_hour}/hour"]
    )
    logger.info("Rate limiting initialized with Redis backend")
except Exception as e:
    # Fallback to in-memory storage
    limiter = Limiter(
        key_func=get_remote_address,
        default_limits=[f"{config.rate_limit.requests_per_hour}/hour"]
    )
    logger.warning(f"Redis not available, using in-memory rate limiting: {e}")


def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    """Custom rate limit exceeded handler."""
    response = HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail={
            "detail": "Rate limit exceeded",
            "error_code": "RATE_LIMIT_EXCEEDED",
            "rate_limit": {
                "limit": exc.detail.get("limit", 0),
                "remaining": exc.detail.get("remaining", 0),
                "reset_time": exc.detail.get("reset_time", 0),
                "retry_after": exc.detail.get("retry_after", 0)
            }
        }
    )
    return response


# Rate limit decorators for different endpoint types
def auth_rate_limit():
    """Rate limit for authentication endpoints."""
    return limiter.limit("5/minute")


def api_rate_limit():
    """Rate limit for general API endpoints."""
    return limiter.limit("100/hour")


def processing_rate_limit():
    """Rate limit for image processing endpoints."""
    return limiter.limit("10/hour")


def status_rate_limit():
    """Rate limit for status check endpoints."""
    return limiter.limit("200/hour")


def admin_rate_limit():
    """Rate limit for admin endpoints."""
    return limiter.limit("50/hour")
