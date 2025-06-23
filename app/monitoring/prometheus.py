"""
Prometheus monitoring setup and metrics.
"""

from prometheus_client import Counter, Histogram, start_http_server
import time

# Initialize Prometheus metrics
REQUEST_COUNT = Counter(
    'api_requests_total', 
    'Total count of API requests', 
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'api_request_latency_seconds', 
    'API request latency in seconds',
    ['method', 'endpoint']
)

PROCESSING_TIME = Histogram(
    'image_processing_time_seconds', 
    'Image processing time in seconds',
    ['operation']
)

JOB_COUNT = Counter(
    'job_count_total', 
    'Total count of jobs', 
    ['status']
)

def setup_prometheus(port=9090):
    """Start Prometheus HTTP server on the specified port."""
    start_http_server(port)
    print(f"Prometheus metrics available on port {port}")

class PrometheusMiddleware:
    """FastAPI middleware for monitoring requests with Prometheus."""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        request_start_time = time.time()
        
        # Extract method and path for Prometheus labels
        method = scope.get("method", "").lower()
        path = scope.get("path", "")
        
        # Store original send function
        original_send = send
        
        # Create a wrapper for send to capture the response status
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                # Extract response status for Prometheus labels
                status = message["status"]
                # Record request count
                REQUEST_COUNT.labels(method=method, endpoint=path, status=status).inc()
                # Record request latency
                REQUEST_LATENCY.labels(method=method, endpoint=path).observe(time.time() - request_start_time)
            
            # Call the original send function
            await original_send(message)
        
        # Call the next middleware with our modified send function
        await self.app(scope, receive, send_wrapper)

def track_processing_time(operation_name):
    """Decorator to track processing time for different operations."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            result = await func(*args, **kwargs)
            PROCESSING_TIME.labels(operation=operation_name).observe(time.time() - start_time)
            return result
        return wrapper
    return decorator

def track_job_status(job_id, status, cache_hit=False):
    """Record job status in Prometheus."""
    JOB_COUNT.labels(status=status).inc()
