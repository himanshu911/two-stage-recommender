"""
Health check API endpoints.

This module defines health check endpoints for monitoring the application.
Design Rationale:
- Comprehensive health checks for all components
- Performance metrics collection
- Database connectivity verification
- Model availability checking
"""

import asyncio
from typing import Dict, Any, List
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, status, Depends

from app.models.schemas import HealthCheckResponse, ErrorResponse
from app.core.dependencies import SessionDep, FeatureServiceDep, RecommendationServiceDep
from app.core.config import get_settings
from app.core.logging import get_logger
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = get_logger(__name__)
router = APIRouter(tags=["Health"])

# Track application start time for accurate uptime calculation
APP_START_TIME = datetime.now(timezone.utc)


@router.get(
    "/",
    response_model=HealthCheckResponse,
    summary="Health check",
    description="Comprehensive health check for all system components"
)
async def health_check(
    session: SessionDep,
    feature_service: FeatureServiceDep,
    recommendation_service: RecommendationServiceDep
) -> HealthCheckResponse:
    """
    Perform comprehensive health check.
    
    Returns:
        Health check response with component status
        
    Raises:
        HTTPException: If critical components are unhealthy
    """
    checks = {}
    timestamp = datetime.now(timezone.utc)
    
    try:
        # Database health check (with 5 second timeout)
        try:
            async with asyncio.timeout(5.0):
                await session.execute(text("SELECT 1"))
            checks["database"] = "healthy"
        except asyncio.TimeoutError:
            checks["database"] = "unhealthy: timeout"
            logger.error("Database health check timed out")
        except Exception as e:
            checks["database"] = f"unhealthy: {str(e)}"
            logger.error("Database health check failed", error=str(e))

        # Feature service health check (with 5 second timeout)
        try:
            async with asyncio.timeout(5.0):
                schema = feature_service.get_feature_schema()
                if schema:
                    checks["feature_service"] = "healthy"
                else:
                    checks["feature_service"] = "unhealthy: no feature schema"
        except asyncio.TimeoutError:
            checks["feature_service"] = "unhealthy: timeout"
            logger.error("Feature service health check timed out")
        except Exception as e:
            checks["feature_service"] = f"unhealthy: {str(e)}"
            logger.error("Feature service health check failed", error=str(e))

        # Recommendation service health check (with 5 second timeout)
        try:
            async with asyncio.timeout(5.0):
                metrics = recommendation_service.get_performance_metrics()
            checks["recommendation_service"] = "healthy"
        except asyncio.TimeoutError:
            checks["recommendation_service"] = "unhealthy: timeout"
            logger.error("Recommendation service health check timed out")
        except Exception as e:
            checks["recommendation_service"] = f"unhealthy: {str(e)}"
            logger.error("Recommendation service health check failed", error=str(e))
        
        # Overall status
        failed_checks = [k for k, v in checks.items() if v != "healthy"]
        overall_status = "healthy" if not failed_checks else "unhealthy"
        
        response = HealthCheckResponse(
            status=overall_status,
            timestamp=timestamp,
            version=get_settings().APP_NAME,
            checks=checks
        )
        
        # Return unhealthy status if critical components failed
        if overall_status == "unhealthy":
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=response.model_dump()  # Convert Pydantic model to dict
            )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Health check failed"
        )


@router.get(
    "/ready",
    response_model=Dict[str, str],
    summary="Readiness check",
    description="Simple readiness check for Kubernetes or load balancers"
)
async def readiness_check(session: SessionDep) -> Dict[str, str]:
    """
    Simple readiness check.
    
    Returns:
        Readiness status
    """
    try:
        # Check database connectivity (with 5 second timeout)
        async with asyncio.timeout(5.0):
            await session.execute(text("SELECT 1"))
        return {"status": "ready", "timestamp": datetime.now(timezone.utc).isoformat()}

    except asyncio.TimeoutError:
        logger.error("Readiness check timed out")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready: timeout"
        )
    except Exception as e:
        logger.error("Readiness check failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready"
        )


@router.get(
    "/live",
    response_model=Dict[str, str],
    summary="Liveness check",
    description="Simple liveness check for Kubernetes"
)
async def liveness_check() -> Dict[str, str]:
    """
    Simple liveness check.
    
    Returns:
        Liveness status
    """
    return {"status": "alive", "timestamp": datetime.now(timezone.utc).isoformat()}


@router.get(
    "/metrics",
    response_model=Dict[str, Any],
    summary="Get application metrics",
    description="Get various application performance and health metrics"
)
async def get_metrics(
    recommendation_service: RecommendationServiceDep
) -> Dict[str, Any]:
    """
    Get application metrics.
    
    Returns:
        Dictionary of application metrics
    """
    try:
        metrics = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": get_settings().APP_NAME,
            "environment": get_settings().ENVIRONMENT,
            "uptime_seconds": (datetime.now(timezone.utc) - APP_START_TIME).total_seconds(),
            "recommendation_metrics": recommendation_service.get_performance_metrics()
        }
        
        return metrics
        
    except Exception as e:
        logger.error("Error getting metrics", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get metrics"
        )


@router.get(
    "/models",
    response_model=Dict[str, Any],
    summary="Get model information",
    description="Get information about ML models and their status"
)
async def get_model_info(
    session: SessionDep
) -> Dict[str, Any]:
    """
    Get ML model information.
    
    Returns:
        Model information dictionary
    """
    try:
        # Query actual ML models from database
        from app.models.database import MLModel
        from sqlalchemy import select

        query = select(MLModel).order_by(MLModel.created_at.desc())
        result = await session.execute(query)
        models = result.scalars().all()

        if not models:
            # Return config-based defaults if no models in database
            return {
                "ranking_model": {
                    "version": get_settings().RANKING_MODEL_VERSION,
                    "status": "not_in_database",
                    "source": "config"
                },
                "candidate_generation": {
                    "embedding_dimension": get_settings().EMBEDDING_DIMENSION,
                    "status": "not_in_database",
                    "source": "config"
                }
            }

        # Group models by type
        models_by_type = {}
        for model in models:
            if model.model_type not in models_by_type:
                models_by_type[model.model_type] = {
                    "version": model.version,
                    "is_active": model.is_active,
                    "created_at": model.created_at.isoformat(),
                    "metrics": model.metrics or {}
                }

        return models_by_type

    except Exception as e:
        logger.error("Error getting model info", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get model information"
        )


@router.get(
    "/dependencies",
    response_model=Dict[str, Any],
    summary="Get dependency information",
    description="Get information about external dependencies"
)
async def get_dependency_info() -> Dict[str, Any]:
    """
    Get dependency information.

    Returns:
        Dependency information dictionary (sanitized, no credentials)
    """
    def sanitize_url(url: str) -> Dict[str, str]:
        """Extract host/port from URL without credentials."""
        from urllib.parse import urlparse
        if not url:
            return {"status": "not configured"}
        try:
            parsed = urlparse(url)
            return {
                "host": parsed.hostname or "unknown",
                "port": parsed.port if parsed.port else "default",
                "scheme": parsed.scheme,
                "status": "configured"
            }
        except Exception:
            return {"status": "invalid"}

    return {
        "database": sanitize_url(get_settings().DATABASE_URL),
        "redis": sanitize_url(get_settings().REDIS_URL),
        "feature_store": sanitize_url(get_settings().FEATURE_STORE_URL),
        "model_storage_path": get_settings().MODEL_STORAGE_PATH,
        "dependencies": {
            "fastapi": "0.104.1",
            "sqlalchemy": "2.0.23",
            "scikit-learn": "1.3.2",
            "faiss": "1.7.4"
        }
    }
