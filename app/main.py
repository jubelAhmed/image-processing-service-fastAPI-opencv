"""
Main application module that defines FastAPI routes and startup/shutdown events.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from datetime import datetime

# Import our modules
from app.monitoring.prometheus import PrometheusMiddleware, setup_prometheus
from app.utils.logging import log_startup_banner, log_processing_step
from app.db.postgres import PostgresClient
from app.routers.facial_processing import router as facial_processing_router
from app.utils.config import config

# Initialize FastAPI app
app = FastAPI(
    title=config.app_name, 
    description="API for processing facial images and generating contour masks",
    version=config.version,
    debug=config.debug
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add Prometheus middleware
app.add_middleware(PrometheusMiddleware)

# Include API routers
app.include_router(facial_processing_router)

# Display startup banner
log_startup_banner("Facial Contour Masking API", "1.0.0")

# Setup Prometheus metrics server and database connections
@app.on_event("startup")
async def startup_event():
    # Start Prometheus metrics server if enabled
    if config.prometheus.enabled:
        setup_prometheus(port=config.prometheus.port)
        log_processing_step("Prometheus metrics server started")
    
    # Initialize PostgreSQL connection if database is enabled
    if config.db.use_database:
        log_processing_step("Initializing PostgreSQL connection...")
        
        app.state.postgres_client = PostgresClient(config.db.connection_string)
        await app.state.postgres_client.connect()
        await app.state.postgres_client.create_tables()
        log_processing_step("PostgreSQL connection established")
    else:
        log_processing_step("Database usage is disabled")

@app.on_event("shutdown")
async def shutdown_event():
    # Close PostgreSQL connection if database was used
    if config.db.use_database and hasattr(app.state, "postgres_client"):
        await app.state.postgres_client.disconnect()
        log_processing_step("PostgreSQL connection closed")

@app.get("/")
async def root():
    return {
        "service": config.app_name,
        "version": config.version,
        "status": "operational",
        "docs_url": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for container orchestration systems."""
    db_status = "connected" if config.db.use_database and hasattr(app.state, "postgres_client") else "disabled"
    
    return {
        "status": "healthy",
        "service": config.app_name,
        "version": config.version,
        "database": db_status,
        "timestamp": str(datetime.now())
    }