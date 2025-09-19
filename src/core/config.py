"""
Global application configuration.
"""

import os
from typing import Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def parse_bool(v: str) -> bool:
    """Parse string to boolean."""
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in ("true", "1", "yes", "y", "t")
    return bool(v)


class DatabaseConfig(BaseSettings):
    """Database connection configuration."""
    use_database: bool = Field(True, description="Enable/disable database usage")
    host: str = "postgres"
    port: int = 5432
    username: str = "postgres"
    password: str = "postgres"
    database: str = "facial_api"
    
    model_config = SettingsConfigDict(
        env_prefix="DB_",
        extra="ignore"
    )
    
    @field_validator('use_database', mode='before')
    @classmethod
    def validate_use_database(cls, v):
        return parse_bool(v)
    
    @property
    def connection_string(self) -> str:
        """Get the database connection string."""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


class PrometheusConfig(BaseSettings):
    """Prometheus monitoring configuration."""
    enabled: bool = Field(True, description="Enable/disable Prometheus metrics")
    port: int = 9090
    
    model_config = SettingsConfigDict(
        env_prefix="PROMETHEUS_",
        extra="ignore"
    )
    
    @field_validator('enabled', mode='before')
    @classmethod
    def validate_enabled(cls, v):
        return parse_bool(v)


class AuthConfig(BaseSettings):
    """Authentication configuration."""
    secret_key: str = Field("your-secret-key-here-change-in-production", description="JWT secret key")
    access_token_expire_minutes: int = Field(30, description="Access token expiration in minutes")
    refresh_token_expire_days: int = Field(7, description="Refresh token expiration in days")
    algorithm: str = Field("HS256", description="JWT algorithm")
    
    model_config = SettingsConfigDict(
        env_prefix="AUTH_",
        extra="ignore"
    )


class RateLimitConfig(BaseSettings):
    """Rate limiting configuration."""
    enabled: bool = Field(True, description="Enable/disable rate limiting")
    requests_per_hour: int = Field(100, description="Requests per hour per IP/user")
    window_seconds: int = Field(3600, description="Rate limit window in seconds")
    burst_limit: int = Field(10, description="Burst limit for short periods")
    
    model_config = SettingsConfigDict(
        env_prefix="RATE_LIMIT_",
        extra="ignore"
    )
    
    @field_validator('enabled', mode='before')
    @classmethod
    def validate_enabled(cls, v):
        return parse_bool(v)


class Config(BaseSettings):
    """Application configuration."""
    app_name: str = "Facial Contour Masking API"
    version: str = "1.0.0"
    debug: bool = Field(False, description="Enable/disable debug mode")
    db: Optional[DatabaseConfig] = None
    auth: Optional[AuthConfig] = None
    prometheus: Optional[PrometheusConfig] = None
    rate_limit: Optional[RateLimitConfig] = None
    
    model_config = SettingsConfigDict(
        env_prefix="APP_",
        extra="ignore"
    )
    
    @field_validator('debug', mode='before')
    @classmethod
    def validate_debug(cls, v):
        return parse_bool(v)


def load_config() -> Config:
    """Load configuration from environment variables."""
    # Determine if we're running in development mode (localhost)
    is_dev = os.getenv("ENV", "development") == "development"
    
    # Set USE_DATABASE environment variable if it's not already set
    if "DB_USE_DATABASE" not in os.environ:
        os.environ["DB_USE_DATABASE"] = "false" if is_dev else "true"
    
    # Set default database host based on environment
    if "DB_HOST" not in os.environ:
        os.environ["DB_HOST"] = "localhost" if is_dev else "postgres"
    
    # Load configs from environment variables
    db_config = DatabaseConfig()
    auth_config = AuthConfig()
    prometheus_config = PrometheusConfig()
    rate_limit_config = RateLimitConfig()
    
    # Create main config
    config = Config()
    config.db = db_config
    config.auth = auth_config
    config.prometheus = prometheus_config
    config.rate_limit = rate_limit_config
    
    return config


# Create a global config instance
config = load_config()
