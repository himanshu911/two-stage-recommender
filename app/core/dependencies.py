"""
Dependency injection configuration for FastAPI.

This module defines dependency functions for FastAPI using the dependency injection
pattern. This enables easy testing and loose coupling between components.

Design Rationale:
- Dependency injection for testability
- Service lifecycle management
- Configuration-based service creation
- Clean separation of concerns

Dependency Graph:
==================

This module acts as the composition root, managing the dependency tree for all services.
FastAPI automatically resolves dependencies from bottom-up (leaf dependencies first).

Complete Dependency Tree:
-------------------------

RecommendationService (Top-level orchestrator)
├── AsyncSession (database session)
├── FeatureService
│   └── AsyncSession
├── CandidateGenerationService
│   ├── AsyncSession
│   ├── FeatureService
│   │   └── AsyncSession (shared/reused)
│   └── candidate_pool_size (from config)
└── RankingService
    ├── AsyncSession
    ├── FeatureService
    │   └── AsyncSession (shared/reused)
    └── model_version (from config)

Repository Dependencies (Simple):
----------------------------------

UserRepository
└── AsyncSession

InteractionRepository
└── AsyncSession

Key Design Points:
------------------
1. **Session Sharing**: FastAPI reuses the same AsyncSession instance across all
   dependencies within a single request (scope="request"). This ensures:
   - Single database transaction per request
   - Efficient connection pool usage
   - Automatic commit/rollback on request completion

2. **Service Composition**: Higher-level services (RecommendationService) depend on
   lower-level services (CandidateGenerationService, RankingService), which in turn
   depend on FeatureService. This creates a clear hierarchy.

3. **Configuration Injection**: Settings are injected via get_settings() calls within
   dependency functions, ensuring configuration changes don't require code changes.

4. **Type Safety**: All dependencies use Annotated types for IDE support and runtime
   validation.

Dependency Resolution Example:
------------------------------
When an endpoint needs RecommendationService:

1. FastAPI sees: rec_service: RecommendationServiceDep
2. Calls: get_recommendation_service()
3. Which needs: session, feature_service, candidate_service, ranking_service
4. For feature_service: Calls get_feature_service()
   - Which needs: session (reuses from step 3)
5. For candidate_service: Calls get_candidate_generation_service()
   - Which needs: session (reuses), feature_service (reuses from step 4)
6. For ranking_service: Calls get_ranking_service()
   - Which needs: session (reuses), feature_service (reuses from step 4)
7. Creates RecommendationService with all resolved dependencies
8. Injects into endpoint function

Total: 1 AsyncSession, 1 FeatureService, shared across all services (efficient!)
"""

from typing import Annotated
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import get_session
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
