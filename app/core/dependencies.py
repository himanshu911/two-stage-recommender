"""
Dependency injection configuration for FastAPI.

This module defines dependency functions for FastAPI using the dependency injection
pattern. This enables easy testing and loose coupling between components.

Design Rationale:
- Dependency injection for testability
- Service lifecycle management
- Configuration-based service creation
- Clean separation of concerns
"""

from typing import AsyncGenerator, Annotated
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_session
from app.core.config import get_settings
from app.services.feature_service import FeatureService
from app.services.candidate_generation import CandidateGenerationService
from app.services.ranking_service import RankingService
from app.services.recommendation_service import RecommendationService
from app.repositories.user_repository import UserRepository
from app.repositories.interaction_repository import InteractionRepository


# Type aliases for dependency injection
SessionDep = Annotated[AsyncSession, Depends(get_session)]


async def get_user_repository(session: SessionDep) -> UserRepository:
    """
    Dependency function to get UserRepository instance.

    Args:
        session: Database session dependency

    Returns:
        UserRepository instance
    """
    return UserRepository(session)


async def get_interaction_repository(session: SessionDep) -> InteractionRepository:
    """
    Dependency function to get InteractionRepository instance.

    Args:
        session: Database session dependency

    Returns:
        InteractionRepository instance
    """
    return InteractionRepository(session)


async def get_feature_service(session: SessionDep) -> FeatureService:
    """
    Dependency function to get FeatureService instance.
    
    Args:
        session: Database session dependency
        
    Returns:
        FeatureService instance
    """
    return FeatureService(session)


async def get_candidate_generation_service(
    session: SessionDep,
    feature_service: Annotated[FeatureService, Depends(get_feature_service)]
) -> CandidateGenerationService:
    """
    Dependency function to get CandidateGenerationService instance.
    
    Args:
        session: Database session dependency
        feature_service: FeatureService dependency
        
    Returns:
        CandidateGenerationService instance
    """
    return CandidateGenerationService(
        session=session,
        feature_service=feature_service,
        candidate_pool_size=get_settings().CANDIDATE_GENERATION_TOP_K
    )


async def get_ranking_service(
    session: SessionDep,
    feature_service: Annotated[FeatureService, Depends(get_feature_service)]
) -> RankingService:
    """
    Dependency function to get RankingService instance.
    
    Args:
        session: Database session dependency
        feature_service: FeatureService dependency
        
    Returns:
        RankingService instance
    """
    return RankingService(
        session=session,
        feature_service=feature_service,
        model_version=get_settings().RANKING_MODEL_VERSION
    )


async def get_recommendation_service(
    session: SessionDep,
    feature_service: Annotated[FeatureService, Depends(get_feature_service)],
    candidate_service: Annotated[CandidateGenerationService, Depends(get_candidate_generation_service)],
    ranking_service: Annotated[RankingService, Depends(get_ranking_service)]
) -> RecommendationService:
    """
    Dependency function to get RecommendationService instance.
    
    Args:
        session: Database session dependency
        feature_service: FeatureService dependency
        candidate_service: CandidateGenerationService dependency
        ranking_service: RankingService dependency
        
    Returns:
        RecommendationService instance
    """
    return RecommendationService(
        session=session,
        feature_service=feature_service,
        candidate_generation_service=candidate_service,
        ranking_service=ranking_service
    )


# Type aliases for repository dependencies
UserRepositoryDep = Annotated[UserRepository, Depends(get_user_repository)]
InteractionRepositoryDep = Annotated[InteractionRepository, Depends(get_interaction_repository)]

# Type aliases for service dependencies
FeatureServiceDep = Annotated[FeatureService, Depends(get_feature_service)]
CandidateServiceDep = Annotated[CandidateGenerationService, Depends(get_candidate_generation_service)]
RankingServiceDep = Annotated[RankingService, Depends(get_ranking_service)]
RecommendationServiceDep = Annotated[RecommendationService, Depends(get_recommendation_service)]
