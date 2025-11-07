"""
Interaction repository implementation with domain-specific methods.

This module extends the base repository with interaction-specific data access patterns.
Design Rationale:
- Specialized queries for interaction data
- Efficient aggregation queries for statistics
- Support for temporal analysis
- Optimized queries for recommendation algorithms
"""

from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta

from sqlalchemy import select, func, and_, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import Interaction, InteractionType, User
from app.repositories.base import SQLModelRepository
from app.core.logging import get_logger

logger = get_logger(__name__)


class InteractionRepository(SQLModelRepository[Interaction]):
    """
    Repository for Interaction entity with domain-specific methods.
    
    Design Rationale:
    - Extends base repository for standard operations
    - Adds interaction-specific analytics methods
    - Optimized queries for ML feature extraction
    - Temporal query support for time-series analysis
    """
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, Interaction)
    
    async def get_user_interactions(
        self,
        user_id: int,
        interaction_type: Optional[InteractionType] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[Interaction]:
        """
        Get interactions for a specific user with optional filtering.
        
        Args:
            user_id: ID of the user
            interaction_type: Optional filter by interaction type
            start_date: Optional filter by start date
            end_date: Optional filter by end date
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of interactions matching the criteria
        """
        try:
            query = select(Interaction).where(Interaction.user_id == user_id)
            
            # Apply filters
            if interaction_type:
                query = query.where(Interaction.interaction_type == interaction_type)
            
            if start_date:
                query = query.where(Interaction.timestamp >= start_date)
            
            if end_date:
                query = query.where(Interaction.timestamp <= end_date)
            
            query = query.order_by(desc(Interaction.timestamp)).offset(skip).limit(limit)
            
            result = await self.session.execute(query)
            interactions = result.scalars().all()
            
            logger.debug(
                "User interactions retrieved",
                user_id=user_id,
                interaction_type=interaction_type,
                count=len(interactions),
                skip=skip,
                limit=limit
            )
            
            return list(interactions)
            
        except Exception as e:
            logger.error(
                "Error retrieving user interactions",
                user_id=user_id,
                interaction_type=interaction_type,
                error=str(e),
                exc_info=True
            )
            raise
    
    async def get_interaction_counts_by_type(
        self,
        user_id: int,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, int]:
        """
        Get interaction counts by type for a user.
        
        Args:
            user_id: ID of the user
            start_date: Optional filter by start date
            end_date: Optional filter by end date
            
        Returns:
            Dictionary mapping interaction types to counts
        """
        try:
            query = (
                select(Interaction.interaction_type, func.count(Interaction.id))
                .where(Interaction.user_id == user_id)
                .group_by(Interaction.interaction_type)
            )
            
            # Apply date filters
            if start_date:
                query = query.where(Interaction.timestamp >= start_date)
            
            if end_date:
                query = query.where(Interaction.timestamp <= end_date)
            
            result = await self.session.execute(query)
            counts = dict(result.all())
            
            logger.debug(
                "Interaction counts retrieved",
                user_id=user_id,
                counts=counts
            )
            
            return counts
            
        except Exception as e:
            logger.error(
                "Error retrieving interaction counts",
                user_id=user_id,
                error=str(e),
                exc_info=True
            )
            raise
    
    async def get_mutual_interactions(
        self,
        user_id: int,
        target_user_id: int
    ) -> List[Interaction]:
        """
        Get mutual interactions between two users.
        
        Args:
            user_id: ID of the first user
            target_user_id: ID of the second user
            
        Returns:
            List of mutual interactions
        """
        try:
            query = (
                select(Interaction)
                .where(
                    and_(
                        Interaction.user_id == user_id,
                        Interaction.target_user_id == target_user_id
                    )
                )
                .union(
                    select(Interaction)
                    .where(
                        and_(
                            Interaction.user_id == target_user_id,
                            Interaction.target_user_id == user_id
                        )
                    )
                )
                .order_by(Interaction.timestamp)
            )
            
            result = await self.session.execute(query)
            interactions = result.scalars().all()
            
            logger.debug(
                "Mutual interactions retrieved",
                user_id=user_id,
                target_user_id=target_user_id,
                count=len(interactions)
            )
            
            return list(interactions)
            
        except Exception as e:
            logger.error(
                "Error retrieving mutual interactions",
                user_id=user_id,
                target_user_id=target_user_id,
                error=str(e),
                exc_info=True
            )
            raise
    
    async def get_recent_interactions(
        self,
        user_id: int,
        days: int = 30,
        limit: int = 100
    ) -> List[Interaction]:
        """
        Get recent interactions for a user.
        
        Args:
            user_id: ID of the user
            days: Number of days to look back
            limit: Maximum number of interactions to return
            
        Returns:
            List of recent interactions
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            query = (
                select(Interaction)
                .where(
                    and_(
                        Interaction.user_id == user_id,
                        Interaction.timestamp >= cutoff_date
                    )
                )
                .order_by(desc(Interaction.timestamp))
                .limit(limit)
            )
            
            result = await self.session.execute(query)
            interactions = result.scalars().all()
            
            logger.debug(
                "Recent interactions retrieved",
                user_id=user_id,
                days=days,
                count=len(interactions)
            )
            
            return list(interactions)
            
        except Exception as e:
            logger.error(
                "Error retrieving recent interactions",
                user_id=user_id,
                days=days,
                error=str(e),
                exc_info=True
            )
            raise
    
    async def get_interaction_timeline(
        self,
        user_id: int,
        interval_days: int = 7,
        lookback_days: int = 30
    ) -> List[Tuple[datetime, str, int]]:
        """
        Get interaction timeline aggregated by time intervals.
        
        Args:
            user_id: ID of the user
            interval_days: Number of days per interval
            lookback_days: Total number of days to look back
            
        Returns:
            List of tuples (date, interaction_type, count)
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=lookback_days)
            
            # This is a simplified version - in production you'd use date_trunc
            query = (
                select(
                    func.date(Interaction.timestamp).label('date'),
                    Interaction.interaction_type,
                    func.count(Interaction.id).label('count')
                )
                .where(
                    and_(
                        Interaction.user_id == user_id,
                        Interaction.timestamp >= cutoff_date
                    )
                )
                .group_by(
                    func.date(Interaction.timestamp),
                    Interaction.interaction_type
                )
                .order_by('date', Interaction.interaction_type)
            )
            
            result = await self.session.execute(query)
            timeline = result.all()
            
            logger.debug(
                "Interaction timeline retrieved",
                user_id=user_id,
                interval_days=interval_days,
                lookback_days=lookback_days,
                count=len(timeline)
            )
            
            return timeline
            
        except Exception as e:
            logger.error(
                "Error retrieving interaction timeline",
                user_id=user_id,
                error=str(e),
                exc_info=True
            )
            raise
    
    async def has_interaction(
        self,
        user_id: int,
        target_user_id: int,
        interaction_type: Optional[InteractionType] = None
    ) -> bool:
        """
        Check if an interaction exists between two users.
        
        Args:
            user_id: ID of the user
            target_user_id: ID of the target user
            interaction_type: Optional specific interaction type to check
            
        Returns:
            True if interaction exists, False otherwise
        """
        try:
            query = (
                select(func.count(Interaction.id))
                .where(
                    and_(
                        Interaction.user_id == user_id,
                        Interaction.target_user_id == target_user_id
                    )
                )
            )
            
            if interaction_type:
                query = query.where(Interaction.interaction_type == interaction_type)
            
            result = await self.session.execute(query)
            count = result.scalar()
            
            return count > 0
            
        except Exception as e:
            logger.error(
                "Error checking interaction existence",
                user_id=user_id,
                target_user_id=target_user_id,
                error=str(e),
                exc_info=True
            )
            raise
    
    async def get_top_interacted_users(
        self,
        user_id: int,
        interaction_type: Optional[InteractionType] = None,
        limit: int = 10
    ) -> List[Tuple[int, int]]:
        """
        Get users most interacted with by the specified user.
        
        Args:
            user_id: ID of the user
            interaction_type: Optional filter by interaction type
            limit: Maximum number of users to return
            
        Returns:
            List of tuples (target_user_id, interaction_count)
        """
        try:
            query = (
                select(
                    Interaction.target_user_id,
                    func.count(Interaction.id).label('interaction_count')
                )
                .where(Interaction.user_id == user_id)
                .group_by(Interaction.target_user_id)
                .order_by(desc('interaction_count'))
                .limit(limit)
            )
            
            if interaction_type:
                query = query.where(Interaction.interaction_type == interaction_type)
            
            result = await self.session.execute(query)
            top_users = result.all()
            
            logger.debug(
                "Top interacted users retrieved",
                user_id=user_id,
                interaction_type=interaction_type,
                count=len(top_users)
            )
            
            return top_users
            
        except Exception as e:
            logger.error(
                "Error retrieving top interacted users",
                user_id=user_id,
                error=str(e),
                exc_info=True
            )
            raise
        