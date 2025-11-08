"""
Repository module initialization.

This module exports repository implementations and provides factory functions.
Design Rationale:
- Centralized repository exports
- Factory pattern for repository creation
- Dependency injection support
- Clean module interface
"""

from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.base import SQLModelRepository
from app.repositories.user_repository import UserRepository
from app.repositories.interaction_repository import InteractionRepository
from app.models.database import User, Interaction, UserEmbedding, UserFeatures, MLModel


__all__ = [
    "SQLModelRepository",
    "UserRepository",
    "InteractionRepository",
    "get_user_repository",
    "get_interaction_repository",
]


def get_user_repository(session: AsyncSession) -> UserRepository:
    """
    Factory function to create a UserRepository instance.
    
    Args:
        session: Database session
        
    Returns:
        UserRepository instance
    """
    return UserRepository(session)


def get_interaction_repository(session: AsyncSession) -> InteractionRepository:
    """
    Factory function to create an InteractionRepository instance.
    
    Args:
        session: Database session
        
    Returns:
        InteractionRepository instance
    """
    return InteractionRepository(session)
