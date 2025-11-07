"""
Recommendation Service that orchestrates the two-stage recommendation pipeline.

This service coordinates between candidate generation and ranking services.
Design Rationale:
- Orchestrates the two-stage recommendation pipeline
- Handles caching and performance optimization
- Provides unified interface for recommendations
- Supports A/B testing and experimentation
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import time

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import User
from app.models.schemas import UserResponse, RecommendationResponse
from app.services.candidate_generation import CandidateGenerationService, Candidate
from app.services.ranking_service import RankingService, RankedCandidate
from app.services.feature_service import FeatureService
from app.repositories.user_repository import UserRepository
from app.core.logging import get_logger
from app.core.config import get_settings

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class RecommendationRequest:
    """Internal request object for recommendations."""
    user_id: int
    limit: int = 20
    exclude_seen: bool = True
    filters: Optional[Dict[str, Any]] = None
    algorithm_version: str = "v1"


@dataclass
class RecommendationMetadata:
    """Metadata about the recommendation generation process."""
    generation_time_ms: float
    candidate_count: int
    ranking_model_version: str
    algorithm_version: str
    timestamp: datetime


class RecommendationService:
    """
    Main service for generating recommendations.
    
    Design Rationale:
    - Orchestrates candidate generation and ranking
    - Implements caching for performance
    - Supports multiple recommendation strategies
    - Provides comprehensive logging and monitoring
    """
    
    def __init__(
        self,
        session: AsyncSession,
        feature_service: FeatureService,
        candidate_generation_service: CandidateGenerationService,
        ranking_service: RankingService
    ):
        self.session = session
        self.feature_service = feature_service
        self.candidate_generation_service = candidate_generation_service
        self.ranking_service = ranking_service
        
        # Initialize repositories
        self.user_repository = UserRepository(session)
        
        # Cache for recommendations (in production, use Redis)
        self._recommendation_cache: Dict[str, Tuple[List[int], datetime]] = {}
        self._cache_ttl = timedelta(minutes=10)
        
        # Performance tracking
        self._performance_metrics: Dict[str, List[float]] = {
            "generation_time": [],
            "candidate_count": [],
            "cache_hit_rate": []
        }
    
    async def get_recommendations(
        self,
        user_id: int,
        limit: int = 20,
        exclude_seen: bool = True,
        filters: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
    ) -> RecommendationResponse:
        """
        Get recommendations for a user.
        
        Args:
            user_id: ID of the user requesting recommendations
            limit: Maximum number of recommendations to return
            exclude_seen: Whether to exclude users already seen
            filters: Additional filters (age_range, location, etc.)
            use_cache: Whether to use cached recommendations
            
        Returns:
            RecommendationResponse with users and metadata
        """
        start_time = time.time()
        
        try:
            # Check cache
            cache_key = f"recs:{user_id}:{limit}:{str(sorted(filters.items()) if filters else '')}"
            
            if use_cache and cache_key in self._recommendation_cache:
                cached_user_ids, timestamp = self._recommendation_cache[cache_key]
                if datetime.utcnow() - timestamp < self._cache_ttl:
                    logger.debug("Recommendations retrieved from cache", user_id=user_id)
                    return await self._build_response_from_cache(
                        user_id, cached_user_ids, start_time
                    )
            
            # Step 1: Generate candidates
            candidates = await self.candidate_generation_service.generate_candidates(
                user_id=user_id,
                limit=settings.CANDIDATE_GENERATION_TOP_K,
                exclude_seen=exclude_seen,
                filters=filters
            )
            
            if not candidates:
                logger.warning("No candidates generated", user_id=user_id)
                return self._empty_response(user_id, start_time)
            
            # Step 2: Rank candidates
            ranked_candidates = await self.ranking_service.rank_candidates(
                user_id=user_id,
                candidate_ids=[c.user_id for c in candidates],
                limit=limit
            )
            
            if not ranked_candidates:
                logger.warning("No candidates ranked", user_id=user_id)
                return self._empty_response(user_id, start_time)
            
            # Step 3: Get user details
            recommended_user_ids = [rc.user_id for rc in ranked_candidates]
            recommended_users = await self.user_repository.get_all(
                limit=len(recommended_user_ids)
            )
            
            # Filter to only recommended users
            recommended_users = [
                user for user in recommended_users
                if user.id in recommended_user_ids
            ]
            
            # Sort by ranking order
            user_map = {user.id: user for user in recommended_users}
            sorted_users = [user_map[user_id] for user_id in recommended_user_ids if user_id in user_map]
            
            # Add ranking scores to user responses
            score_map = {rc.user_id: rc.score for rc in ranked_candidates}
            
            # Build response
            generation_time_ms = (time.time() - start_time) * 1000
            
            response = RecommendationResponse(
                recommendations=[
                    UserResponse(
                        id=user.id,
                        name=user.name,
                        age=user.age,
                        gender=user.gender,
                        location=user.location,
                        bio=user.bio,
                        interests=user.interests,
                        created_at=user.created_at,
                        last_active_at=user.last_active_at,
                        match_score=score_map.get(user.id, 0.0)
                    )
                    for user in sorted_users
                ],
                total_count=len(sorted_users),
                algorithm_version=self.ranking_service.model_version,
                generation_time_ms=generation_time_ms
            )
            
            # Cache the response
            if use_cache:
                self._recommendation_cache[cache_key] = (
                    recommended_user_ids,
                    datetime.utcnow()
                )
            
            # Track performance
            self._track_performance(len(sorted_users), generation_time_ms, use_cache)
            
            logger.info(
                "Recommendations generated",
                user_id=user_id,
                count=len(response.recommendations),
                generation_time_ms=generation_time_ms
            )
            
            return response
            
        except Exception as e:
            logger.error("Error generating recommendations", user_id=user_id, error=str(e))
            return self._empty_response(user_id, start_time)
    
    async def get_recommendations_with_explanation(
        self,
        user_id: int,
        limit: int = 20
    ) -> Tuple[List[UserResponse], List[Dict[str, Any]]]:
        """
        Get recommendations with explanations for why each user was recommended.
        
        Args:
            user_id: ID of the user requesting recommendations
            limit: Maximum number of recommendations
            
        Returns:
            Tuple of (user_responses, explanations)
        """
        try:
            # Get recommendations
            response = await self.get_recommendations(user_id, limit)
            
            # Generate explanations
            explanations = []
            for user_response in response.recommendations:
                explanation = await self._generate_explanation(user_id, user_response)
                explanations.append(explanation)
            
            return response.recommendations, explanations
            
        except Exception as e:
            logger.error("Error generating recommendations with explanation", user_id=user_id, error=str(e))
            return [], []
    
    async def _generate_explanation(
        self,
        user_id: int,
        recommended_user: UserResponse
    ) -> Dict[str, Any]:
        """Generate explanation for why a user was recommended."""
        try:
            # Get features for both users
            user_features = await self.feature_service.get_features(user_id)
            candidate_features = await self.feature_service.get_features(recommended_user.id)
            
            explanation = {
                "user_id": recommended_user.id,
                "match_score": recommended_user.match_score,
                "reasons": []
            }
            
            # Age similarity
            age_diff = abs(user_features.get("age", 0) - recommended_user.age)
            if age_diff <= 2:
                explanation["reasons"].append("Similar age")
            
            # Location
            if user_features.get("location") == recommended_user.location:
                explanation["reasons"].append("Same location")
            
            # Common interests
            user_interests = set(user_features.get("interests", []))
            candidate_interests = set(recommended_user.interests)
            common_interests = user_interests & candidate_interests
            
            if common_interests:
                explanation["reasons"].append(f"Common interests: {', '.join(list(common_interests)[:3])}")
            
            # Activity level
            if candidate_features.get("recent_activity_30d", 0) > 5:
                explanation["reasons"].append("Active user")
            
            # Collaborative filtering explanation
            if recommended_user.match_score > 0.7:
                explanation["reasons"].append("Highly compatible based on your preferences")
            
            return explanation
            
        except Exception as e:
            logger.error("Error generating explanation", user_id=user_id, error=str(e))
            return {"user_id": recommended_user.id, "reasons": ["Based on your preferences"]}
    
    async def _build_response_from_cache(
        self,
        user_id: int,
        user_ids: List[int],
        start_time: float
    ) -> RecommendationResponse:
        """Build response from cached user IDs."""
        try:
            # Get user details
            users = await self.user_repository.get_all(limit=len(user_ids))
            users = [user for user in users if user.id in user_ids]
            
            # Sort by cached order
            user_map = {user.id: user for user in users}
            sorted_users = [user_map[user_id] for user_id in user_ids if user_id in user_map]
            
            generation_time_ms = (time.time() - start_time) * 1000
            
            return RecommendationResponse(
                recommendations=[
                    UserResponse(
                        id=user.id,
                        name=user.name,
                        age=user.age,
                        gender=user.gender,
                        location=user.location,
                        bio=user.bio,
                        interests=user.interests,
                        created_at=user.created_at,
                        last_active_at=user.last_active_at,
                        match_score=0.5  # Default score for cached results
                    )
                    for user in sorted_users
                ],
                total_count=len(sorted_users),
                algorithm_version=self.ranking_service.model_version,
                generation_time_ms=generation_time_ms
            )
            
        except Exception as e:
            logger.error("Error building response from cache", user_id=user_id, error=str(e))
            return self._empty_response(user_id, start_time)
    
    def _empty_response(self, user_id: int, start_time: float) -> RecommendationResponse:
        """Return empty recommendation response."""
        generation_time_ms = (time.time() - start_time) * 1000
        
        return RecommendationResponse(
            recommendations=[],
            total_count=0,
            algorithm_version=self.ranking_service.model_version,
            generation_time_ms=generation_time_ms
        )
    
    def _track_performance(
        self,
        candidate_count: int,
        generation_time_ms: float,
        cache_used: bool
    ) -> None:
        """Track performance metrics."""
        self._performance_metrics["generation_time"].append(generation_time_ms)
        self._performance_metrics["candidate_count"].append(candidate_count)
        
        # Keep only recent metrics
        for key in self._performance_metrics:
            self._performance_metrics[key] = self._performance_metrics[key][-100:]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for monitoring."""
        return {
            "avg_generation_time_ms": sum(self._performance_metrics["generation_time"]) / len(self._performance_metrics["generation_time"]),
            "avg_candidate_count": sum(self._performance_metrics["candidate_count"]) / len(self._performance_metrics["candidate_count"]),
            "cache_size": len(self._recommendation_cache),
            "model_version": self.ranking_service.model_version
        }
    
    async def refresh_cache(self) -> None:
        """Refresh recommendation cache by clearing old entries."""
        current_time = datetime.utcnow()
        
        # Remove expired cache entries
        expired_keys = [
            key for key, (_, timestamp) in self._recommendation_cache.items()
            if current_time - timestamp > self._cache_ttl
        ]
        
        for key in expired_keys:
            del self._recommendation_cache[key]
        
        logger.info("Cache refreshed", removed_entries=len(expired_keys))
    
    async def update_algorithm_version(self, version: str) -> None:
        """Update the algorithm version."""
        await self.ranking_service.update_model_version(version)
        
        # Clear cache to force regeneration with new algorithm
        self._recommendation_cache.clear()
        
        logger.info("Algorithm version updated", version=version)
        