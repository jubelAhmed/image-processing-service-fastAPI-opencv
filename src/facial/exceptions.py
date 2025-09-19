"""
Facial processing specific exceptions.
"""

from fastapi import HTTPException, status
from src.facial.constants import ProcessingErrorCode


class ProcessingException(HTTPException):
    """Base processing exception."""
    
    def __init__(self, detail: str, error_code: ProcessingErrorCode):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "detail": detail,
                "error_code": error_code.value
            }
        )


class NoFaceDetectedException(ProcessingException):
    """Raised when no face is detected in the image."""
    
    def __init__(self, detail: str = "No face detected in the image"):
        super().__init__(detail, ProcessingErrorCode.NO_FACE_DETECTED)


class InvalidImageException(ProcessingException):
    """Raised when image data is invalid or corrupted."""
    
    def __init__(self, detail: str = "Invalid image data provided"):
        super().__init__(detail, ProcessingErrorCode.INVALID_IMAGE)


class InvalidInputException(ProcessingException):
    """Raised when input parameters are invalid."""
    
    def __init__(self, detail: str = "Invalid input parameters"):
        super().__init__(detail, ProcessingErrorCode.INVALID_INPUT)


class ProcessingErrorException(ProcessingException):
    """Raised when image processing fails."""
    
    def __init__(self, detail: str = "Image processing failed"):
        super().__init__(detail, ProcessingErrorCode.PROCESSING_ERROR)


class DatabaseException(HTTPException):
    """Raised when database operations fail."""
    
    def __init__(self, detail: str = "Database operation failed"):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "detail": detail,
                "error_code": ProcessingErrorCode.DATABASE_ERROR.value
            }
        )


class CacheException(HTTPException):
    """Raised when cache operations fail."""
    
    def __init__(self, detail: str = "Cache operation failed"):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "detail": detail,
                "error_code": ProcessingErrorCode.CACHE_ERROR.value
            }
        )
