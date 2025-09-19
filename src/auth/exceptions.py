"""
Authentication-specific exceptions.
"""

from fastapi import HTTPException, status
from src.auth.constants import AuthErrorCode


class AuthException(HTTPException):
    """Base authentication exception."""
    
    def __init__(self, detail: str, error_code: AuthErrorCode):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "detail": detail,
                "error_code": error_code.value
            },
            headers={"WWW-Authenticate": "Bearer"}
        )


class InvalidCredentialsException(AuthException):
    """Raised when credentials are invalid."""
    
    def __init__(self):
        super().__init__(
            "Invalid username or password",
            AuthErrorCode.INVALID_CREDENTIALS
        )


class TokenExpiredException(AuthException):
    """Raised when token has expired."""
    
    def __init__(self):
        super().__init__(
            "Token has expired",
            AuthErrorCode.TOKEN_EXPIRED
        )


class TokenInvalidException(AuthException):
    """Raised when token is invalid."""
    
    def __init__(self):
        super().__init__(
            "Invalid token",
            AuthErrorCode.TOKEN_INVALID
        )


class UserNotFoundException(HTTPException):
    """Raised when user is not found."""
    
    def __init__(self):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "detail": "User not found",
                "error_code": AuthErrorCode.USER_NOT_FOUND.value
            }
        )


class UserInactiveException(AuthException):
    """Raised when user account is inactive."""
    
    def __init__(self):
        super().__init__(
            "User account is inactive",
            AuthErrorCode.USER_INACTIVE
        )


class InsufficientPermissionsException(HTTPException):
    """Raised when user lacks required permissions."""
    
    def __init__(self):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "detail": "Insufficient permissions",
                "error_code": AuthErrorCode.INSUFFICIENT_PERMISSIONS.value
            }
        )


class PasswordTooWeakException(HTTPException):
    """Raised when password doesn't meet requirements."""
    
    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "detail": detail,
                "error_code": AuthErrorCode.PASSWORD_TOO_WEAK.value
            }
        )


class UsernameTakenException(HTTPException):
    """Raised when username is already taken."""
    
    def __init__(self):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "detail": "Username already taken",
                "error_code": AuthErrorCode.USERNAME_TAKEN.value
            }
        )


class EmailTakenException(HTTPException):
    """Raised when email is already taken."""
    
    def __init__(self):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "detail": "Email already taken",
                "error_code": AuthErrorCode.EMAIL_TAKEN.value
            }
        )
