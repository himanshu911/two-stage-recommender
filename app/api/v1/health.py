"""
Health check API endpoints.

This module defines health check endpoints for monitoring the application.
Design Rationale:
- Comprehensive health checks for all components
- Performance metrics collection
- Database connectivity verification
- Model availability checking
"""

from typing import Dict, Any, List
from datetime import datetime
from fastapi import APIRouter, HTTPException, status, Depends

from app.models.schemas import HealthCheckResponse, ErrorResponse
from app.core.dependencies import SessionDep, FeatureServiceDep, RecommendationServiceDep
from app.core.config import get_settings
from app.core.logging import get_logger
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = get_logger(__name__)
router = APIRouter()


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
    timestamp = datetime.utcnow()
    
    try:
        # Database health check
        try:
            await session.execute(text("SELECT 1"))
            checks["database"] = "healthy"
        except Exception as e:
            checks["database"] = f"unhealthy: {str(e)}"
            logger.error("Database health check failed", error=str(e))
        
        # Feature service health check
        try:
            schema = feature_service.get_feature_schema()
            if schema:
                checks["feature_service"] = "healthy"
            else:
                checks["feature_service"] = "unhealthy: no feature schema"
        except Exception as e:
            checks["feature_service"] = f"unhealthy: {str(e)}"
            logger.error("Feature service health check failed", error=str(e))
        
        # Recommendation service health check
        try:
            metrics = recommendation_service.get_performance_metrics()
            checks["recommendation_service"] = "healthy"
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
                detail=response
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
        # Check database connectivity
        await session.execute(text("SELECT 1"))
        return {"status": "ready", "timestamp": datetime.utcnow().isoformat()}
        
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
    return {"status": "alive", "timestamp": datetime.utcnow().isoformat()}


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
            "timestamp": datetime.utcnow().isoformat(),
            "version": get_settings().APP_NAME,
            "environment": get_settings().ENVIRONMENT,
            "uptime_seconds": (datetime.utcnow() - datetime(2024, 1, 1)).total_seconds(),
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
        # In a real implementation, this would query the ML models table
        # For now, return placeholder information
        return {
            "ranking_model": {
                "version": get_settings().RANKING_MODEL_VERSION,
                "status": "active",
                "last_updated": datetime.utcnow().isoformat()
            },
            "candidate_generation": {
                "embedding_dimension": get_settings().EMBEDDING_DIMENSION,
                "status": "active",
                "last_trained": datetime.utcnow().isoformat()
            }
        }
        
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
        Dependency information dictionary
    """
    return {
        "database_url": get_settings().DATABASE_URL,
        "redis_url": get_settings().REDIS_URL,
        "model_storage_path": get_settings().MODEL_STORAGE_PATH,
        "feature_store_url": get_settings().FEATURE_STORE_URL,
        "dependencies": {
            "fastapi": "0.104.1",
            "sqlalchemy": "2.0.23",
            "scikit-learn": "1.3.2",
            "faiss": "1.7.4"
        }
    }
