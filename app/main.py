"""
Main FastAPI application module.

This module creates and configures the FastAPI application instance.
Design Rationale:
- Factory pattern for app creation
- Comprehensive middleware configuration
- Error handling and logging setup
- CORS and security configuration
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.core.database import create_tables
from app.api.v1 import api_router
from app.models.schemas import ErrorResponse

# Get settings and configure logging
settings = get_settings()
configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    This function handles application startup and shutdown events.
    
    Design Rationale:
    - Async context manager for proper resource management
    - Database initialization on startup
    - Graceful shutdown handling
    - Comprehensive logging of lifecycle events
    """
    # Startup
    logger.info("Application starting up", app_name=settings.APP_NAME)
    
    try:
        # Create database tables
        await create_tables()
        logger.info("Database tables created/verified")
        
    except Exception as e:
        logger.error("Failed to initialize database", error=str(e))
        raise
    
    yield  # Application is running
    
    # Shutdown
    logger.info("Application shutting down")


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application.
    
    Returns:
        Configured FastAPI application instance
    """
    app = FastAPI(
        title=settings.APP_NAME,
        description="Production-ready dating recommender system with ML-powered recommendations",
        version="1.0.0",
        docs_url="/docs" if settings.ENVIRONMENT != "production" else None,
        redoc_url="/redoc" if settings.ENVIRONMENT != "production" else None,
        openapi_url="/openapi.json" if settings.ENVIRONMENT != "production" else None,
        lifespan=lifespan
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add GZip compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Add request ID middleware
    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        """Add request ID to all requests for tracing."""
        from uuid import uuid4
        request_id = str(uuid4())
        
        # Add request ID to context
        from app.core.logging import LoggingContext
        with LoggingContext(request_id=request_id):
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response
    
    # Add timing middleware
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        """Add processing time header to responses."""
        import time
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response
    
    return app


# Create the application instance
app = create_app()


# Custom exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors."""
    logger.warning("Request validation error", path=request.url.path, errors=exc.errors())
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error="Request validation failed",
            detail=str(exc.errors()),
            request_id=request.headers.get("X-Request-ID")
        ).model_dump(mode='json')
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions."""
    logger.warning("HTTP exception", path=request.url.path, status_code=exc.status_code, detail=exc.detail)
    
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            request_id=request.headers.get("X-Request-ID")
        ).model_dump(mode='json')
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error("Unhandled exception", path=request.url.path, error=str(exc), exc_info=True)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            detail="An unexpected error occurred",
            request_id=request.headers.get("X-Request-ID")
        ).model_dump(mode='json')
    )


# Include API routes
app.include_router(api_router, prefix=settings.API_V1_STR)


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with application information."""
    return {
        "name": settings.APP_NAME,
        "version": "1.0.0",
        "description": "Production-ready dating recommender system",
        "docs": "/docs" if settings.ENVIRONMENT != "production" else None,
        "health": "/api/v1/health"
    }


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Application startup event."""
    logger.info("Application startup complete", environment=settings.ENVIRONMENT)


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event."""
    logger.info("Application shutdown initiated")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        workers=settings.WORKERS,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
    