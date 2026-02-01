"""
User API endpoints.

This module defines the REST API endpoints for user management.
Design Rationale:
- RESTful API design with proper HTTP methods
- Request/response validation with Pydantic
- Comprehensive error handling
- Proper status codes and error responses
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, status, Depends, Query

from app.models.sql_models import User
from app.models.schemas import (
    UserCreateRequest,
    UserUpdateRequest,
    UserResponse,
    ErrorResponse
)
from app.repositories.user_repository import UserRepository
from app.core.dependencies import UserRepositoryDep
from app.core.logging import get_logger
from app.core.config import get_settings

logger = get_logger(__name__)
router = APIRouter(tags=["Users"])


@router.post(
    "/",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new user",
    description="Create a new user account with profile information"
)
async def create_user(
    user_data: UserCreateRequest,
    user_repository: UserRepositoryDep
) -> UserResponse:
    """
    Create a new user.
    
    Args:
        user_data: User creation data
        
    Returns:
        Created user response
        
    Raises:
        HTTPException: If user creation fails
    """
    try:
        # Create user entity
        user = User(**user_data.model_dump())
        
        # Save to database
        created_user = await user_repository.create(user)
        
        logger.info("User created", user_id=created_user.id, name=created_user.name)
        
        return UserResponse.model_validate(created_user)
        
    except Exception as e:
        logger.error("Error creating user", error=str(e), user_data=user_data.model_dump())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user"
        )


@router.get(
    "/{user_id}",
    response_model=UserResponse,
    summary="Get user by ID",
    description="Retrieve a user's profile information by ID"
)
async def get_user(
    user_id: int,
    user_repository: UserRepositoryDep
) -> UserResponse:
    """
    Get user by ID.
    
    Args:
        user_id: User ID
        
    Returns:
        User response
        
    Raises:
        HTTPException: If user not found
    """
    try:
        user = await user_repository.get_by_id(user_id)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with ID {user_id} not found"
            )
        
        return UserResponse.model_validate(user)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting user", user_id=user_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user"
        )


@router.put(
    "/{user_id}",
    response_model=UserResponse,
    summary="Update user",
    description="Update a user's profile information"
)
async def update_user(
    user_id: int,
    user_data: UserUpdateRequest,
    user_repository: UserRepositoryDep
) -> UserResponse:
    """
    Update user.
    
    Args:
        user_id: User ID
        user_data: Update data (partial updates supported)
        
    Returns:
        Updated user response
        
    Raises:
        HTTPException: If user not found or update fails
    """
    try:
        # Filter out None values for partial update
        update_data = {
            key: value for key, value in user_data.model_dump(exclude_unset=True).items()
            if value is not None
        }
        
        if not update_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid fields to update"
            )
        
        updated_user = await user_repository.update(user_id, update_data)
        
        if not updated_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with ID {user_id} not found"
            )
        
        logger.info("User updated", user_id=user_id, updated_fields=list(update_data.keys()))
        
        return UserResponse.model_validate(updated_user)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error updating user", user_id=user_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user"
        )


@router.delete(
    "/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete user",
    description="Delete a user account"
)
async def delete_user(
    user_id: int,
    user_repository: UserRepositoryDep
) -> None:
    """
    Delete user.
    
    Args:
        user_id: User ID
        
    Raises:
        HTTPException: If user not found or deletion fails
    """
    try:
        success = await user_repository.delete(user_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with ID {user_id} not found"
            )
        
        logger.info("User deleted", user_id=user_id)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error deleting user", user_id=user_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete user"
        )


@router.get(
    "/search/active",
    response_model=List[UserResponse],
    summary="Search active users",
    description="Get users who have been active recently"
)
async def search_active_users(
    user_repository: UserRepositoryDep,
    days_since_active: int = Query(30, ge=1, le=365, description="Days since last activity"),
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records to return")
) -> List[UserResponse]:
    """
    Search for active users.
    
    Args:
        days_since_active: Number of days to consider as "active"
        skip: Number of records to skip
        limit: Maximum number of records to return
        
    Returns:
        List of active users
    """
    try:
        users = await user_repository.get_active_users(days_since_active, skip, limit)
        
        logger.debug(
            "Active users retrieved",
            days_since_active=days_since_active,
            count=len(users)
        )
        
        return [UserResponse.model_validate(user) for user in users]
        
    except Exception as e:
        logger.error("Error searching active users", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to search active users"
        )


@router.get(
    "/search/by-interest",
    response_model=List[UserResponse],
    summary="Search users by interest",
    description="Find users who share specific interests"
)
async def search_by_interest(
    interest: str,
    user_repository: UserRepositoryDep,
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records to return")
) -> List[UserResponse]:
    """
    Search users by interest.
    
    Args:
        interest: Interest to search for
        skip: Number of records to skip
        limit: Maximum number of records to return
        
    Returns:
        List of users with the specified interest
    """
    try:
        users = await user_repository.get_users_with_interest(interest, skip, limit)
        
        logger.debug("Users retrieved by interest", interest=interest, count=len(users))
        
        return [UserResponse.model_validate(user) for user in users]
        
    except Exception as e:
        logger.error("Error searching by interest", interest=interest, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to search users by interest"
        )


@router.get(
    "/",
    response_model=List[UserResponse],
    summary="List users",
    description="Get a list of users with pagination"
)
async def list_users(
    user_repository: UserRepositoryDep,
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records to return"),
    location: Optional[str] = Query(None, description="Filter by location"),
    min_age: Optional[int] = Query(None, ge=18, description="Minimum age filter"),
    max_age: Optional[int] = Query(None, le=120, description="Maximum age filter")
) -> List[UserResponse]:
    """
    List users with optional filtering.

    Args:
        skip: Number of records to skip
        limit: Maximum number of records to return
        location: Optional location filter
        min_age: Optional minimum age filter
        max_age: Optional maximum age filter

    Returns:
        List of user responses
    """
    try:
        # Validate age range
        if min_age is not None and max_age is not None:
            if min_age > max_age:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"min_age ({min_age}) cannot be greater than max_age ({max_age})"
                )

        # Build filters
        filters = {}
        if location:
            filters["location"] = location
        if min_age is not None:
            filters["min_age"] = min_age
        if max_age is not None:
            filters["max_age"] = max_age

        # Apply filters if provided, otherwise use basic pagination
        if filters:
            if location:
                users = await user_repository.get_by_location(location, skip, limit)
            elif min_age is not None or max_age is not None:
                users = await user_repository.get_by_age_range(
                    min_age or 18,
                    max_age or 120,
                    skip,
                    limit
                )
            else:
                users = await user_repository.get_all(skip, limit)
        else:
            users = await user_repository.get_all(skip, limit)

        logger.debug("Users listed", count=len(users), skip=skip, limit=limit)

        return [UserResponse.model_validate(user) for user in users]

    except Exception as e:
        logger.error("Error listing users", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list users"
        )
