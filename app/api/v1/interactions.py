"""
Interaction API endpoints.

This module defines the REST API endpoints for user interactions (likes, dislikes, etc.).
Design Rationale:
- RESTful design with proper HTTP methods
- Comprehensive interaction tracking
- Real-time ML feature updates
- Proper error handling and validation
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from fastapi import APIRouter, HTTPException, status, Depends, Query

from app.models.sql_models import Interaction, InteractionType
from app.models.schemas import (
    InteractionResponse,
    ErrorResponse,
    TimelineEntry
)
from app.core.dependencies import (
    UserRepositoryDep,
    InteractionRepositoryDep,
    FeatureServiceDep,
    SessionDep
)
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["Interactions"])


@router.post(
    "/",
    response_model=InteractionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create interaction",
    description="Record a user interaction (like, dislike, super_like, block)"
)
async def create_interaction(
    interaction_data: InteractionCreateRequest,
    user_repository: UserRepositoryDep,
    interaction_repository: InteractionRepositoryDep,
    feature_service: FeatureServiceDep,
    session: SessionDep
) -> InteractionResponse:
    """
    Create a new interaction.
    
    Args:
        interaction_data: Interaction data
        
    Returns:
        Created interaction response
        
    Raises:
        HTTPException: If interaction creation fails
    """
    try:
        # Prevent self-interaction
        if interaction_data.user_id == interaction_data.target_user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot create interaction with yourself"
            )

        # Validate that both users exist
        user = await user_repository.get_by_id(interaction_data.user_id)
        target_user = await user_repository.get_by_id(interaction_data.target_user_id)
        
        if not user:
             raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with ID {interaction_data.user_id} not found"
            )

        if not target_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Target user with ID {interaction_data.target_user_id} not found"
            )
        
        # Create interaction entity
        interaction = Interaction(
            user_id=interaction_data.user_id,
            target_user_id=interaction_data.target_user_id,
            interaction_type=interaction_data.interaction_type,
            context=interaction_data.context or {}
        )

        # Save to database (no commit yet)
        created_interaction = await interaction_repository.create(interaction, commit=False)

        # Update user's last active timestamp (no commit yet)
        await user_repository.update_last_active(interaction_data.user_id, commit=False)

        # Update ML features (compute first, if it fails we haven't committed DB yet)
        # Note: In a real prod environment, this should be a background task. 
        # But if we want strong consistency for the "feature update" part, we do it here.
        # However, the requirement was to fix the DB atomicity.
        try:
            await feature_service.compute_and_store_features(
                interaction_data.user_id,
                version="v1"
            )
        except Exception as e:
            # If features fail, we log but still might want to proceed with the interaction
            # OR we might want to rollback everything. 
            # Given the audit report said: "If step 2 or 3 fails, the interaction exists but user state is inconsistent."
            # We will choose to proceed (log warning) but ensure DB consistency is at least atomic.
             logger.warning("Failed to update user features", error=str(e))
        
        # Commit the transaction for both DB operations
        await session.commit()
        await session.refresh(created_interaction)
        
        logger.info(
            "Interaction created",
            user_id=created_interaction.user_id,
            target_user_id=created_interaction.target_user_id,
            interaction_type=created_interaction.interaction_type
        )
        
        return InteractionResponse.model_validate(created_interaction)
        
    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        logger.error("Error creating interaction", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create interaction"
        )


@router.get(
    "/user/{user_id}",
    response_model=List[InteractionResponse],
    summary="Get user interactions",
    description="Get all interactions made by a specific user"
)
async def get_user_interactions(
    user_id: int,
    interaction_repository: InteractionRepositoryDep,
    user_repository: UserRepositoryDep,
    interaction_type: Optional[InteractionType] = Query(None, description="Filter by interaction type"),
    start_date: Optional[datetime] = Query(None, description="Filter by start date"),
    end_date: Optional[datetime] = Query(None, description="Filter by end date"),
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records to return")
) -> List[InteractionResponse]:
    """
    Get interactions for a user.
    
    Args:
        user_id: User ID
        interaction_type: Optional filter by interaction type
        start_date: Optional filter by start date
        end_date: Optional filter by end date
        skip: Number of records to skip
        limit: Maximum number of records to return
        
    Returns:
        List of interaction responses
    """
    try:
        # Validate date range
        if start_date is not None and end_date is not None:
            if start_date > end_date:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"start_date ({start_date}) cannot be after end_date ({end_date})"
                )

        # Validate user exists
        user = await user_repository.get_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with ID {user_id} not found"
            )

        interactions = await interaction_repository.get_user_interactions(
            user_id=user_id,
            interaction_type=interaction_type,
            start_date=start_date,
            end_date=end_date,
            skip=skip,
            limit=limit
        )
        
        logger.debug(
            "User interactions retrieved",
            user_id=user_id,
            count=len(interactions)
        )
        
        return [InteractionResponse.model_validate(interaction) for interaction in interactions]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting user interactions", user_id=user_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve interactions"
        )


@router.get(
    "/stats/{user_id}",
    response_model=Dict[str, int],
    summary="Get interaction statistics",
    description="Get interaction counts by type for a user"
)
async def get_interaction_stats(
    user_id: int,
    interaction_repository: InteractionRepositoryDep,
    user_repository: UserRepositoryDep,
    start_date: Optional[datetime] = Query(None, description="Filter by start date"),
    end_date: Optional[datetime] = Query(None, description="Filter by end date")
) -> Dict[str, int]:
    """
    Get interaction statistics for a user.
    
    Args:
        user_id: User ID
        start_date: Optional filter by start date
        end_date: Optional filter by end date
        
    Returns:
        Dictionary of interaction counts by type
    """
    try:
        # Validate date range
        if start_date is not None and end_date is not None:
            if start_date > end_date:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"start_date ({start_date}) cannot be after end_date ({end_date})"
                )

        # Validate user exists
        user = await user_repository.get_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with ID {user_id} not found"
            )

        stats = await interaction_repository.get_interaction_counts_by_type(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date
        )
        
        logger.debug("Interaction statistics retrieved", user_id=user_id, stats=stats)
        
        return stats
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting interaction statistics", user_id=user_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve interaction statistics"
        )


@router.get(
    "/mutual/{user_id}/{target_user_id}",
    response_model=List[InteractionResponse],
    summary="Get mutual interactions",
    description="Get all interactions between two users"
)
async def get_mutual_interactions(
    user_id: int,
    target_user_id: int,
    interaction_repository: InteractionRepositoryDep,
    user_repository: UserRepositoryDep
) -> List[InteractionResponse]:
    """
    Get mutual interactions between two users.
    
    Args:
        user_id: First user ID
        target_user_id: Second user ID
        
    Returns:
        List of mutual interactions
    """
    try:
        # Validate both users exist
        user1 = await user_repository.get_by_id(user_id)
        user2 = await user_repository.get_by_id(target_user_id)
        
        if not user1 or not user2:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="One or both users not found"
            )
        
        interactions = await interaction_repository.get_mutual_interactions(
            user_id=user_id,
            target_user_id=target_user_id
        )
        
        logger.debug(
            "Mutual interactions retrieved",
            user_id=user_id,
            target_user_id=target_user_id,
            count=len(interactions)
        )
        
        return [InteractionResponse.model_validate(interaction) for interaction in interactions]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Error getting mutual interactions",
            user_id=user_id,
            target_user_id=target_user_id,
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve mutual interactions"
        )


@router.get(
    "/recent/{user_id}",
    response_model=List[InteractionResponse],
    summary="Get recent interactions",
    description="Get recent interactions for a user within specified days"
)
async def get_recent_interactions(
    user_id: int,
    interaction_repository: InteractionRepositoryDep,
    user_repository: UserRepositoryDep,
    days: int = Query(30, ge=1, le=365, description="Number of days to look back"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of interactions")
) -> List[InteractionResponse]:
    """
    Get recent interactions for a user.
    
    Args:
        user_id: User ID
        days: Number of days to look back
        limit: Maximum number of interactions
        
    Returns:
        List of recent interactions
    """
    try:
        # Validate user exists
        user = await user_repository.get_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with ID {user_id} not found"
            )
        
        interactions = await interaction_repository.get_recent_interactions(
            user_id=user_id,
            days=days,
            limit=limit
        )
        
        logger.debug(
            "Recent interactions retrieved",
            user_id=user_id,
            days=days,
            count=len(interactions)
        )
        
        return [InteractionResponse.model_validate(interaction) for interaction in interactions]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Error getting recent interactions",
            user_id=user_id,
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve recent interactions"
        )


@router.get(
    "/timeline/{user_id}",
    response_model=List[TimelineEntry],
    summary="Get interaction timeline",
    description="Get interaction counts over time"
)
async def get_interaction_timeline(
    user_id: int,
    interaction_repository: InteractionRepositoryDep,
    user_repository: UserRepositoryDep,
    interval_days: int = Query(7, ge=1, le=30, description="Days per interval"),
    lookback_days: int = Query(30, ge=1, le=365, description="Total days to look back")
) -> List[TimelineEntry]:
    """
    Get interaction timeline for a user.
    
    Args:
        user_id: User ID
        interval_days: Days per interval
        lookback_days: Total days to look back
        
    Returns:
        List of interaction counts by date and type
    """
    try:
        # Validate user exists
        user = await user_repository.get_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with ID {user_id} not found"
            )
        
        timeline = await interaction_repository.get_interaction_timeline(
            user_id=user_id,
            interval_days=interval_days,
            lookback_days=lookback_days
        )
        
        # Convert to list of TimelineEntry
        timeline_data = []
        for date, interaction_type, count in timeline:
            timeline_data.append(TimelineEntry(
                date=date.isoformat(),
                interaction_type=interaction_type,
                count=count
            ))
        
        logger.debug(
            "Interaction timeline retrieved",
            user_id=user_id,
            intervals=len(timeline_data)
        )
        
        return timeline_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Error getting interaction timeline",
            user_id=user_id,
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve interaction timeline"
        )
    