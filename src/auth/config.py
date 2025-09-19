"""
Authentication configuration settings.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AuthConfig(BaseSettings):
    """Authentication configuration."""
    secret_key: str = Field("your-secret-key-change-in-production", description="JWT secret key")
    access_token_expire_minutes: int = Field(60, description="Access token expiry in minutes")
    refresh_token_expire_days: int = Field(30, description="Refresh token expiry in days")
    algorithm: str = Field("HS256", description="JWT algorithm")
    
    model_config = SettingsConfigDict(
        env_prefix="AUTH_",
        extra="ignore"
    )
