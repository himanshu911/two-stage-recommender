"""
Recommendation API endpoints.

This module defines the REST API endpoints for getting recommendations.
Design Rationale:
- Clean API interface for recommendation requests
- Comprehensive filtering and pagination support
- Performance monitoring and caching
- Detailed recommendation explanations
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, status, Depends, Query

from app.models.schemas import (
    UserResponse,
    RecommendationResponse,
    RecommendationRequest,
    ExplanationResponse,
    PerformanceMetrics,
    ExplanationEntry
)
from sqlalchemy import select

from app.models.database import MLModel
from app.services.recommendation_service import RecommendationService
from app.core.dependencies import RecommendationServiceDep, FeatureServiceDep, SessionDep
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["Recommendations"])


@router.get(
    "/users/{user_id}/recommendations",
    response_model=RecommendationResponse,
    summary="Get user recommendations",
    description="Get personalized recommendations for a user"
)
async def get_recommendations(
    user_id: int,
    recommendation_service: RecommendationServiceDep,
    session: SessionDep,
    limit: int = Query(20, ge=1, le=100, description="Number of recommendations"),
    exclude_seen: bool = Query(True, description="Exclude users already seen"),
    min_age: Optional[int] = Query(None, ge=18, description="Minimum age filter"),
    max_age: Optional[int] = Query(None, le=120, description="Maximum age filter"),
    location: Optional[str] = Query(None, description="Location filter"),
    algorithm_version: Optional[str] = Query(None, description="Algorithm version to use")
) -> RecommendationResponse:
    """
    Get personalized recommendations for a user.
    
    Args:
        user_id: ID of the user requesting recommendations
        limit: Number of recommendations to return
        exclude_seen: Whether to exclude users already seen
        min_age: Optional minimum age filter
        max_age: Optional maximum age filter
        location: Optional location filter
        algorithm_version: Optional algorithm version to use
        
    Returns:
        Recommendation response with users and metadata
        
    Raises:
        HTTPException: If recommendations cannot be generated
    """
    try:
        # Validate algorithm version if specified
        if algorithm_version:
            # Get available versions
            query = select(MLModel.version).distinct()
            result = await session.execute(query)
            available_versions = [r[0] for r in result.all()]
            if not available_versions:
                available_versions = ["v1.0.0"]

            if algorithm_version not in available_versions:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Invalid algorithm version '{algorithm_version}'. Available versions: {', '.join(available_versions)}"
                )

        # Build filters
        filters = {}
        if min_age is not None:
            filters["min_age"] = min_age
        if max_age is not None:
            filters["max_age"] = max_age
        if location:
            filters["location"] = location

        # Update algorithm version if specified
        if algorithm_version:
            await recommendation_service.update_algorithm_version(algorithm_version)
        
        # Get recommendations
        response = await recommendation_service.get_recommendations(
            user_id=user_id,
            limit=limit,
            exclude_seen=exclude_seen,
            filters=filters
        )
        
        logger.info(
            "Recommendations generated",
            user_id=user_id,
            count=len(response.recommendations),
            algorithm_version=response.algorithm_version
        )
        
        return response
        
    except Exception as e:
        logger.error("Error generating recommendations", user_id=user_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate recommendations"
        )


@router.get(
    "/users/{user_id}/recommendations/explain",
    response_model=ExplanationResponse,
    summary="Get recommendations with explanations",
    description="Get recommendations with detailed explanations for each recommendation"
)
async def get_recommendations_with_explanations(
    user_id: int,
    recommendation_service: RecommendationServiceDep,
    limit: int = Query(20, ge=1, le=100, description="Number of recommendations")
) -> ExplanationResponse:
    """
    Get recommendations with explanations.
    
    Args:
        user_id: ID of the user requesting recommendations
        limit: Number of recommendations to return
        
    Returns:
        Dictionary containing recommendations and explanations
    """
    try:
        recommendations, explanations = await recommendation_service.get_recommendations_with_explanation(
            user_id=user_id,
            limit=limit
        )
        
        # Transform explanations to match ExplanationEntry schema if needed
        # Assuming explanations is a dict of user_id -> explanation details
        typed_explanations = {}
        for uid, expl in explanations.items():
            typed_explanations[uid] = ExplanationEntry(
                user_id=int(uid) if isinstance(uid, str) else uid,
                score=expl.get("score", 0.0),
                reason=expl.get("reason", []) if isinstance(expl.get("reason"), list) else [str(expl.get("reason"))],
                contributing_features=expl.get("contributing_features", {})
            )

        return ExplanationResponse(
            recommendations=recommendations,
            explanations=typed_explanations,
            total_count=len(recommendations)
        )
        
    except Exception as e:
        logger.error(
            "Error generating recommendations with explanations",
            user_id=user_id,
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate recommendations with explanations"
        )


@router.get(
    "/users/{user_id}/similar",
    response_model=List[UserResponse],
    summary="Find similar users",
    description="Find users similar to the specified user"
)
async def find_similar_users(
    user_id: int,
    feature_service: FeatureServiceDep,
    limit: int = Query(20, ge=1, le=100, description="Number of similar users")
) -> List[UserResponse]:
    """
    Find users similar to the specified user.
    
    Args:
        user_id: ID of the user
        limit: Number of similar users to return
        
    Returns:
        List of similar users
    """
    # This endpoint is not yet implemented
    # Will use collaborative filtering model when ready
    logger.info("Similar users endpoint called but not implemented", user_id=user_id)
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Similar users feature is not yet implemented"
    )


@router.post(
    "/refresh",
    status_code=status.HTTP_200_OK,
    summary="Refresh recommendation cache",
    description="Manually refresh the recommendation cache"
)
async def refresh_recommendations(
    recommendation_service: RecommendationServiceDep
) -> Dict[str, str]:
    """
    Refresh recommendation cache.
    
    Returns:
        Success message
    """
    try:
        await recommendation_service.refresh_cache()
        return {"message": "Recommendation cache refreshed successfully"}
        
    except Exception as e:
        logger.error("Error refreshing recommendation cache", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to refresh recommendation cache"
        )


@router.get(
    "/performance",
    response_model=PerformanceMetrics,
    summary="Get recommendation performance metrics",
    description="Get performance metrics for the recommendation system"
)
async def get_performance_metrics(
    recommendation_service: RecommendationServiceDep
) -> PerformanceMetrics:
    """
    Get recommendation performance metrics.
    
    Returns:
        Performance metrics dictionary
    """
    try:
        metrics = recommendation_service.get_performance_metrics()
        return PerformanceMetrics(**metrics)
        
    except Exception as e:
        logger.error("Error getting performance metrics", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get performance metrics"
        )


@router.get(
    "/algorithm/versions",
    response_model=List[str],
    summary="Get available algorithm versions",
    description="List all available recommendation algorithm versions"
)
async def get_algorithm_versions(session: SessionDep) -> List[str]:
    """
    Get available algorithm versions from the database.

    Returns:
        List of available algorithm versions
    """
    try:
        query = select(MLModel.version).distinct()
        result = await session.execute(query)
        versions = [r[0] for r in result.all()]

        if not versions:
            # Return 404 if no models in database
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No algorithm versions found in database"
            )

        return versions

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error fetching algorithm versions", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch algorithm versions"
        )
