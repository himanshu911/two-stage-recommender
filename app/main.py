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
from sqlalchemy.exc import IntegrityError
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
    - Graceful shutdown handling with try/finally
    - Comprehensive logging of lifecycle events
    - Startup failures prevent app start, shutdown errors are logged but non-fatal
    """
    # Startup logic
    logger.info("Application starting", app_name=settings.APP_NAME,
    environment=settings.ENVIRONMENT)

    try:
        # Create database tables
        await create_tables()
        logger.info("Database tables created/verified")

        # Initialize other resources (ML models, caches, connections) like this:
        # await load_ml_models()
        # await init_cache()

    except Exception as e:
        logger.exception("Startup failed", error=str(e))
        raise  # Critical: prevent app from starting with broken state

    try:
        yield  # Application is running
    finally:
        # Shutdown logic - always runs if startup succeeded
        logger.info("Application shutting down")

        # Clean up resources (nested try/except prevents shutdown failures)
        try:
            # Add cleanup tasks here as needed:
            # await close_db_connections()
            # await cleanup_ml_models()
            # await close_cache_connections()
            pass
        except Exception as e:
            logger.exception("Error during shutdown cleanup", error=str(e))
            # Don't raise - allow graceful shutdown to complete  


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
        """
        Add request ID to all requests for distributed tracing.

        Accepts X-Request-ID from upstream services (e.g., API gateway, backend)
        or generates new one if not present. This enables request tracing across
        service boundaries.
        """
        from uuid import uuid4

        # Accept from upstream if present, otherwise generate
        request_id = request.headers.get("X-Request-ID") or str(uuid4())

        # Store in request state for access in exception handlers
        request.state.request_id = request_id

        # Add request ID to logging context
        from app.core.logging import LoggingContext
        with LoggingContext(request_id=request_id):
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response
    
    # Add timing middleware
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        """
        Measure and log request processing time.

        Adds X-Process-Time header for client-side monitoring and logs
        duration for aggregation, alerting, and P95/P99 analysis.
        """
        import time
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time

        # Add header for client monitoring
        response.headers["X-Process-Time"] = str(process_time)

        # Log for observability (metrics aggregation, alerting)
        logger.info(
            "Request completed",
            path=request.url.path,
            method=request.method,
            status_code=response.status_code,
            duration_ms=round(process_time * 1000, 2)
        )

        return response
    
    return app


# Create the application instance
app = create_app()


# Custom exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors."""
    request_id = getattr(request.state, "request_id", None)

    logger.warning("Request validation error", path=request.url.path, errors=exc.errors(), request_id=request_id)

    response = JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error="Request validation failed",
            detail=exc.errors(),  # Structured detail for client parsing
            request_id=request_id
        ).model_dump(mode='json')
    )

    # Always set request ID header
    if request_id:
        response.headers["X-Request-ID"] = request_id

    return response


@app.exception_handler(IntegrityError)
async def integrity_exception_handler(request: Request, exc: IntegrityError):
    """Handle database integrity errors (e.g., unique constraints)."""
    request_id = getattr(request.state, "request_id", None)

    logger.warning("Integrity error", path=request.url.path, error=str(exc), request_id=request_id)

    response = JSONResponse(
        status_code=status.HTTP_409_CONFLICT,
        content=ErrorResponse(
            error="Conflict",
            detail="Database constraint violation",
            request_id=request_id
        ).model_dump(mode='json')
    )

    if request_id:
        response.headers["X-Request-ID"] = request_id

    return response


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions."""
    request_id = getattr(request.state, "request_id", None)

    # Log level based on status code: warning for <500, error for >=500
    log_func = logger.error if exc.status_code >= 500 else logger.warning
    log_func("HTTP exception", path=request.url.path, status_code=exc.status_code, detail=exc.detail, request_id=request_id)

    response = JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            request_id=request_id
        ).model_dump(mode='json')
    )

    # Always set request ID header
    if request_id:
        response.headers["X-Request-ID"] = request_id

    return response


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    request_id = getattr(request.state, "request_id", None)

    # Use logger.exception() to automatically capture traceback
    logger.exception("Unhandled exception", path=request.url.path, error=str(exc), request_id=request_id)

    response = JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            detail="An unexpected error occurred",
            request_id=request_id
        ).model_dump(mode='json')
    )

    # Always set request ID header
    if request_id:
        response.headers["X-Request-ID"] = request_id

    return response


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
    