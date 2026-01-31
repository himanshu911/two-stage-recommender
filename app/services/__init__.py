"""
Service module initialization.

This module exports service implementations.
Design Rationale:
- Centralized service exports
- Clean module interface
- Consistent with repositories pattern
- Public API documentation via __all__

Note: Service construction/dependency injection is handled in
app/core/dependencies.py, not here. This module only provides
convenient imports.
"""

from app.services.feature_service import FeatureService
from app.services.candidate_generation import (
    CandidateGenerationService,
    Candidate
)
from app.services.ranking_service import (
    RankingService,
    RankedCandidate
)
from app.services.recommendation_service import RecommendationService

__all__ = [
    # Services
    "FeatureService",
    "CandidateGenerationService",
    "RankingService",
    "RecommendationService",

    # Data classes
    "Candidate",
    "RankedCandidate",
]
