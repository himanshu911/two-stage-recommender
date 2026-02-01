"""
User repository implementation with domain-specific methods.

This module extends the base repository with user-specific data access patterns.
Design Rationale:
- Extends base repository for common operations
- Adds domain-specific query methods
- Implements repository pattern for testability
- Optimized queries for common user operations
"""

from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime, timedelta, timezone

from sqlalchemy import select, func, and_, or_, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.sql_models import User, Interaction, InteractionType
from app.repositories.base import SQLModelRepository
from app.core.logging import get_logger

logger = get_logger(__name__)


class UserRepository(SQLModelRepository[User]):
    """
    Repository for User entity with domain-specific methods.

    Design Rationale:
    - Extends base repository for standard CRUD operations
    - Adds user-specific query methods for common patterns
    - Implements efficient queries with proper indexing
    - Supports filtering and pagination for large datasets
    """

    def __init__(self, session: AsyncSession):
        super().__init__(session, User)

    async def get_by_ids(self, ids: List[int]) -> List[User]:
        """
        Get users by list of IDs - efficient batch query.

        Args:
            ids: List of user IDs to fetch

        Returns:
            List of users matching the IDs
        """
        if not ids:
            return []

        try:
            query = select(User).where(User.id.in_(ids))
            result = await self.session.execute(query)
            users = result.scalars().all()

            logger.debug(
                "Users retrieved by IDs",
                requested_ids=len(ids),
                found=len(users)
            )

            return list(users)

        except Exception as e:
            logger.error(
                "Error retrieving users by IDs",
                ids=ids,
                error=str(e),
                exc_info=True
            )
            raise

    async def get_by_location(
        self,
        location: str,
        skip: int = 0,
        limit: int = 100
    ) -> List[User]:
        """
        Get users by location with pagination.
        
        Args:
            location: Location to filter by
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of users in the specified location
        """
        try:
            query = (
                select(User)
                .where(User.location.ilike(f"%{location}%"))
                .offset(skip)
                .limit(limit)
            )
            
            result = await self.session.execute(query)
            users = result.scalars().all()
            
            logger.debug(
                "Users retrieved by location",
                location=location,
                count=len(users),
                skip=skip,
                limit=limit
            )
            
            return list(users)
            
        except Exception as e:
            logger.error(
                "Error retrieving users by location",
                location=location,
                error=str(e),
                exc_info=True
            )
            raise
    
    async def get_by_age_range(
        self,
        min_age: int,
        max_age: int,
        skip: int = 0,
        limit: int = 100
    ) -> List[User]:
        """
        Get users within an age range.
        
        Args:
            min_age: Minimum age (inclusive)
            max_age: Maximum age (inclusive)
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of users within the age range
        """
        try:
            query = (
                select(User)
                .where(and_(User.age >= min_age, User.age <= max_age))
                .offset(skip)
                .limit(limit)
            )
            
            result = await self.session.execute(query)
            users = result.scalars().all()
            
            logger.debug(
                "Users retrieved by age range",
                min_age=min_age,
                max_age=max_age,
                count=len(users),
                skip=skip,
                limit=limit
            )
            
            return list(users)
            
        except Exception as e:
            logger.error(
                "Error retrieving users by age range",
                min_age=min_age,
                max_age=max_age,
                error=str(e),
                exc_info=True
            )
            raise
    
    async def get_active_users(
        self,
        days_since_active: int = 30,
        skip: int = 0,
        limit: int = 100
    ) -> List[User]:
        """
        Get users active within the specified number of days.
        
        Args:
            days_since_active: Number of days to consider as "active"
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of active users
        """
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_since_active)
            
            query = (
                select(User)
                .where(
                    or_(
                        User.last_active_at >= cutoff_date,
                        User.created_at >= cutoff_date
                    )
                )
                .offset(skip)
                .limit(limit)
            )
            
            result = await self.session.execute(query)
            users = result.scalars().all()
            
            logger.debug(
                "Active users retrieved",
                days_since_active=days_since_active,
                count=len(users),
                skip=skip,
                limit=limit
            )
            
            return list(users)
            
        except Exception as e:
            logger.error(
                "Error retrieving active users",
                days_since_active=days_since_active,
                error=str(e),
                exc_info=True
            )
            raise
    
    async def get_users_with_interest(
        self,
        interest: str,
        skip: int = 0,
        limit: int = 100
    ) -> List[User]:
        """
        Get users who have a specific interest.
        
        Args:
            interest: Interest to search for
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of users with the specified interest
        """
        try:
            query = (
                select(User)
                .where(User.interests.contains([interest]))
                .offset(skip)
                .limit(limit)
            )
            
            result = await self.session.execute(query)
            users = result.scalars().all()
            
            logger.debug(
                "Users retrieved by interest",
                interest=interest,
                count=len(users),
                skip=skip,
                limit=limit
            )
            
            return list(users)
            
        except Exception as e:
            logger.error(
                "Error retrieving users by interest",
                interest=interest,
                error=str(e),
                exc_info=True
            )
            raise
    
    async def get_users_not_interacted_with(
        self,
        user_id: int,
        skip: int = 0,
        limit: int = 100
    ) -> List[User]:
        """
        Get users that the specified user hasn't interacted with.
        
        Args:
            user_id: ID of the user
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of users not yet interacted with
        """
        try:
            # Subquery to get users that have been interacted with
            interacted_subquery = (
                select(Interaction.target_user_id)
                .where(Interaction.user_id == user_id)
            )
            
            query = (
                select(User)
                .where(
                    and_(
                        User.id != user_id,
                        ~User.id.in_(interacted_subquery)
                    )
                )
                .offset(skip)
                .limit(limit)
            )
            
            result = await self.session.execute(query)
            users = result.scalars().all()
            
            logger.debug(
                "Users not interacted with retrieved",
                user_id=user_id,
                count=len(users),
                skip=skip,
                limit=limit
            )
            
            return list(users)
            
        except Exception as e:
            logger.error(
                "Error retrieving users not interacted with",
                user_id=user_id,
                error=str(e),
                exc_info=True
            )
            raise
    
    async def get_interaction_statistics(self, user_id: int) -> Dict[str, int]:
        """
        Get interaction statistics for a user.
        
        Args:
            user_id: ID of the user
            
        Returns:
            Dictionary with interaction counts by type
        """
        try:
            query = (
                select(Interaction.interaction_type, func.count(Interaction.id))
                .where(Interaction.user_id == user_id)
                .group_by(Interaction.interaction_type)
            )
            
            result = await self.session.execute(query)
            statistics = dict(result.all())
            
            logger.debug(
                "Interaction statistics retrieved",
                user_id=user_id,
                statistics=statistics
            )
            
            return statistics
            
        except Exception as e:
            logger.error(
                "Error retrieving interaction statistics",
                user_id=user_id,
                error=str(e),
                exc_info=True
            )
            raise
    
    async def update_last_active(self, user_id: int, commit: bool = True) -> bool:
        """
        Update user's last active timestamp.
        
        Args:
            user_id: ID of the user
            commit: Whether to commit the transaction immediately
            
        Returns:
            True if update was successful
        """
        try:
            query = (
                update(User)
                .where(User.id == user_id)
                .values(last_active_at=datetime.now(timezone.utc))
            )
            
            result = await self.session.execute(query)
            if commit:
                await self.session.commit()
            else:
                await self.session.flush()
            
            updated = result.rowcount > 0
            
            logger.debug(
                "User last active timestamp updated",
                user_id=user_id,
                updated=updated
            )
            
            return updated
            
        except Exception as e:
            await self.session.rollback()
            logger.error(
                "Error updating user last active timestamp",
                user_id=user_id,
                error=str(e),
                exc_info=True
            )
            raise
        