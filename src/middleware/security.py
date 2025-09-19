"""
Security middleware for headers and CORS.
"""

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response as StarletteResponse
from src.core.utils import logger


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers."""
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> StarletteResponse:
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        
        # Content Security Policy
        csp = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self'; "
            "connect-src 'self'; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self'"
        )
        response.headers["Content-Security-Policy"] = csp
        
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for enhanced request logging."""
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> StarletteResponse:
        # Log request
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        
        logger.info(
            f"Request: {request.method} {request.url.path} "
            f"from {client_ip} ({user_agent})"
        )
        
        # Process request
        response = await call_next(request)
        
        # Log response
        logger.info(
            f"Response: {request.method} {request.url.path} "
            f"→ {response.status_code}"
        )
        
        return response


class CORSSecurityMiddleware(BaseHTTPMiddleware):
    """Enhanced CORS middleware with security considerations."""
    
    def __init__(self, app, allowed_origins: list = None, allowed_methods: list = None):
        super().__init__(app)
        self.allowed_origins = allowed_origins or ["*"]
        self.allowed_methods = allowed_methods or ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> StarletteResponse:
        # Handle preflight requests
        if request.method == "OPTIONS":
            response = StarletteResponse()
            response.headers["Access-Control-Allow-Origin"] = self._get_origin(request)
            response.headers["Access-Control-Allow-Methods"] = ", ".join(self.allowed_methods)
            response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With"
            response.headers["Access-Control-Max-Age"] = "86400"  # 24 hours
            return response
        
        # Process request
        response = await call_next(request)
        
        # Add CORS headers
        origin = self._get_origin(request)
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Expose-Headers"] = "X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset"
        
        return response
    
    def _get_origin(self, request: Request) -> str:
        """Get allowed origin for the request."""
        origin = request.headers.get("origin")
        
        if "*" in self.allowed_origins:
            return origin or "*"
        
        if origin in self.allowed_origins:
            return origin
        
        return self.allowed_origins[0] if self.allowed_origins else "*"
