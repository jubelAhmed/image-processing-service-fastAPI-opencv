"""
Authentication utility functions.
"""

import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any
from passlib.context import CryptContext
from jose import jwt
from src.auth.config import AuthConfig
from src.auth.constants import TokenType
from src.auth.exceptions import TokenExpiredException, TokenInvalidException

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Get auth config
auth_config = AuthConfig()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)


def create_access_token(data: Dict[str, Any], expires_delta: timedelta = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=auth_config.access_token_expire_minutes)
    
    to_encode.update({"exp": expire, "type": TokenType.ACCESS.value})
    encoded_jwt = jwt.encode(to_encode, auth_config.secret_key, algorithm=auth_config.algorithm)
    return encoded_jwt


def create_refresh_token(data: Dict[str, Any]) -> str:
    """Create a JWT refresh token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=auth_config.refresh_token_expire_days)
    to_encode.update({"exp": expire, "type": TokenType.REFRESH.value})
    encoded_jwt = jwt.encode(to_encode, auth_config.secret_key, algorithm=auth_config.algorithm)
    return encoded_jwt


def verify_token(token: str, token_type: TokenType = TokenType.ACCESS) -> Dict[str, Any]:
    """Verify and decode a JWT token."""
    try:
        payload = jwt.decode(token, auth_config.secret_key, algorithms=[auth_config.algorithm])
        
        # Check token type
        if payload.get("type") != token_type.value:
            raise TokenInvalidException()
        
        # Check expiration
        exp = payload.get("exp")
        if exp is None or datetime.utcnow() > datetime.fromtimestamp(exp):
            raise TokenExpiredException()
        
        return payload
        
    except jwt.JWTError:
        raise TokenInvalidException()


def generate_token_hash(token: str) -> str:
    """Generate a hash for storing refresh tokens."""
    return hashlib.sha256(token.encode()).hexdigest()


def generate_random_string(length: int = 32) -> str:
    """Generate a random string for various purposes."""
    return secrets.token_urlsafe(length)


def validate_password_strength(password: str) -> tuple[bool, str]:
    """Validate password strength and return (is_valid, error_message)."""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    
    if not any(c.isupper() for c in password):
        return False, "Password must contain at least one uppercase letter"
    
    if not any(c.islower() for c in password):
        return False, "Password must contain at least one lowercase letter"
    
    if not any(c.isdigit() for c in password):
        return False, "Password must contain at least one digit"
    
    return True, ""


def validate_username(username: str) -> tuple[bool, str]:
    """Validate username format and return (is_valid, error_message)."""
    if len(username) < 3:
        return False, "Username must be at least 3 characters long"
    
    if len(username) > 50:
        return False, "Username must be less than 50 characters"
    
    if not username.isalnum():
        return False, "Username must contain only alphanumeric characters"
    
    return True, ""
