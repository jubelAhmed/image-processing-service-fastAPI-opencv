"""
Constants for facial processing module.
"""

from enum import Enum


class JobStatus(str, Enum):
    """Job status enumeration."""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ProcessingErrorCode(str, Enum):
    """Processing error codes."""
    NO_FACE_DETECTED = "NO_FACE_DETECTED"
    INVALID_IMAGE = "INVALID_IMAGE"
    PROCESSING_ERROR = "PROCESSING_ERROR"
    INVALID_INPUT = "INVALID_INPUT"
    DATABASE_ERROR = "DATABASE_ERROR"
    CACHE_ERROR = "CACHE_ERROR"


class GeneratorType(str, Enum):
    """Output generator types."""
    SVG = "svg"
    PNG = "png"
    JSON = "json"


class RegionType(int, Enum):
    """Facial region types."""
    FOREHEAD = 1
    LEFT_EYE = 2
    RIGHT_EYE = 3
    NOSE = 4
    LEFT_CHEEK = 5
    RIGHT_CHEEK = 6
    MOUTH = 7


# Region name mapping
REGION_NAMES = {
    RegionType.FOREHEAD: "forehead",
    RegionType.LEFT_EYE: "left_eye",
    RegionType.RIGHT_EYE: "right_eye",
    RegionType.NOSE: "nose",
    RegionType.LEFT_CHEEK: "left_cheek",
    RegionType.RIGHT_CHEEK: "right_cheek",
    RegionType.MOUTH: "mouth"
}

# Rate limiting constants
RATE_LIMITS = {
    "processing": "10/hour",
    "status": "200/hour",
    "auth": "5/minute",
    "admin": "50/hour"
}
