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
    RecommendationRequest
)
from app.services.recommendation_service import RecommendationService
from app.core.dependencies import RecommendationServiceDep, FeatureServiceDep
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get(
    "/users/{user_id}/recommendations",
    response_model=RecommendationResponse,
    summary="Get user recommendations",
    description="Get personalized recommendations for a user"
)
async def get_recommendations(
    user_id: int,
    recommendation_service: RecommendationServiceDep,
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
    response_model=Dict[str, Any],
    summary="Get recommendations with explanations",
    description="Get recommendations with detailed explanations for each recommendation"
)
async def get_recommendations_with_explanations(
    user_id: int,
    recommendation_service: RecommendationServiceDep,
    limit: int = Query(20, ge=1, le=100, description="Number of recommendations")
) -> Dict[str, Any]:
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
        
        return {
            "recommendations": [rec.model_dump() for rec in recommendations],
            "explanations": explanations,
            "total_count": len(recommendations)
        }
        
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
    try:
        # This would typically use the collaborative filtering model
        # For now, return empty list as placeholder
        logger.warning("Similar users endpoint not fully implemented", user_id=user_id)
        return []
        
    except Exception as e:
        logger.error("Error finding similar users", user_id=user_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to find similar users"
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
    response_model=Dict[str, Any],
    summary="Get recommendation performance metrics",
    description="Get performance metrics for the recommendation system"
)
async def get_performance_metrics(
    recommendation_service: RecommendationServiceDep
) -> Dict[str, Any]:
    """
    Get recommendation performance metrics.
    
    Returns:
        Performance metrics dictionary
    """
    try:
        return recommendation_service.get_performance_metrics()
        
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
async def get_algorithm_versions() -> List[str]:
    """
    Get available algorithm versions.
    
    Returns:
        List of available algorithm versions
    """
    # In a real implementation, this would query the database
    # For now, return hardcoded versions
    return ["v1.0.0", "v1.1.0", "v2.0.0-beta"]
