"""
Global exception classes.
"""

from fastapi import HTTPException, status


class BaseAPIException(HTTPException):
    """Base API exception with error code support."""
    
    def __init__(self, detail: str, error_code: str = None, status_code: int = status.HTTP_400_BAD_REQUEST):
        super().__init__(
            status_code=status_code,
            detail={
                "detail": detail,
                "error_code": error_code
            }
        )


class ValidationException(BaseAPIException):
    """Raised when input validation fails."""
    
    def __init__(self, detail: str = "Validation error"):
        super().__init__(detail, "VALIDATION_ERROR")


class NotFoundException(BaseAPIException):
    """Raised when a resource is not found."""
    
    def __init__(self, detail: str = "Resource not found"):
        super().__init__(detail, "NOT_FOUND", status.HTTP_404_NOT_FOUND)


class ConflictException(BaseAPIException):
    """Raised when there's a conflict with the current state."""
    
    def __init__(self, detail: str = "Conflict"):
        super().__init__(detail, "CONFLICT", status.HTTP_409_CONFLICT)


class UnauthorizedException(BaseAPIException):
    """Raised when authentication is required."""
    
    def __init__(self, detail: str = "Unauthorized"):
        super().__init__(detail, "UNAUTHORIZED", status.HTTP_401_UNAUTHORIZED)


class ForbiddenException(BaseAPIException):
    """Raised when access is forbidden."""
    
    def __init__(self, detail: str = "Forbidden"):
        super().__init__(detail, "FORBIDDEN", status.HTTP_403_FORBIDDEN)


class InternalServerException(BaseAPIException):
    """Raised when an internal server error occurs."""
    
    def __init__(self, detail: str = "Internal server error"):
        super().__init__(detail, "INTERNAL_SERVER_ERROR", status.HTTP_500_INTERNAL_SERVER_ERROR)


class DatabaseError(BaseAPIException):
    """Raised when a database error occurs."""
    
    def __init__(self, detail: str = "Database error"):
        super().__init__(detail, "DATABASE_ERROR", status.HTTP_500_INTERNAL_SERVER_ERROR)
