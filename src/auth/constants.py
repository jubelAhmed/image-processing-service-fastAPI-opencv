"""
Authentication constants and enums.
"""

from enum import Enum


class TokenType(str, Enum):
    """JWT token types."""
    ACCESS = "access"
    REFRESH = "refresh"


class UserRole(str, Enum):
    """User roles."""
    USER = "user"
    SUPERUSER = "superuser"


class AuthErrorCode(str, Enum):
    """Authentication error codes."""
    INVALID_CREDENTIALS = "INVALID_CREDENTIALS"
    TOKEN_EXPIRED = "TOKEN_EXPIRED"
    TOKEN_INVALID = "TOKEN_INVALID"
    USER_NOT_FOUND = "USER_NOT_FOUND"
    USER_INACTIVE = "USER_INACTIVE"
    INSUFFICIENT_PERMISSIONS = "INSUFFICIENT_PERMISSIONS"
    PASSWORD_TOO_WEAK = "PASSWORD_TOO_WEAK"
    USERNAME_TAKEN = "USERNAME_TAKEN"
    EMAIL_TAKEN = "EMAIL_TAKEN"
