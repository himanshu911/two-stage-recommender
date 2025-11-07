"""
Feature Service for centralized feature management.

This service implements the feature store pattern for ML feature management.
Design Rationale:
- Centralized feature computation and storage
- Feature versioning for reproducibility
- Real-time and batch feature support
- Training-serving consistency
"""

from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import json
from abc import ABC, abstractmethod

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, desc

from app.models.database import User, Interaction, UserFeatures, UserEmbedding
from app.repositories.user_repository import UserRepository
from app.repositories.interaction_repository import InteractionRepository
from app.core.logging import get_logger

logger = get_logger(__name__)


class FeatureExtractor(ABC):
    """
    Abstract base class for feature extractors.
    
    Design Rationale:
    - Abstract interface for different feature types
    - Supports both real-time and batch extraction
    - Type hints for feature schemas
    - Extensible for new feature types
    """
    
    @abstractmethod
    async def extract(self, user_id: int, **context: Any) -> Dict[str, Any]:
        """Extract features for a user."""
        pass
    
    @abstractmethod
    def get_feature_schema(self) -> Dict[str, str]:
        """Get the schema of features this extractor produces."""
        pass


class DemographicFeatureExtractor(FeatureExtractor):
    """Extractor for demographic features."""
    
    def __init__(self, user_repository: UserRepository):
        self.user_repository = user_repository
    
    async def extract(self, user_id: int, **context: Any) -> Dict[str, Any]:
        """Extract demographic features for a user."""
        user = await self.user_repository.get_by_id(user_id)
        if not user:
            return {}
        
        return {
            "age": user.age,
            "gender": user.gender,
            "location": user.location,
            "account_age_days": (datetime.utcnow() - user.created_at).days,
            "interests_count": len(user.interests)
        }
    
    def get_feature_schema(self) -> Dict[str, str]:
        return {
            "age": "numeric",
            "gender": "categorical",
            "location": "categorical",
            "account_age_days": "numeric",
            "interests_count": "numeric"
        }


class BehavioralFeatureExtractor(FeatureExtractor):
    """Extractor for behavioral features."""
    
    def __init__(self, interaction_repository: InteractionRepository):
        self.interaction_repository = interaction_repository
    
    async def extract(self, user_id: int, **context: Any) -> Dict[str, Any]:
        """Extract behavioral features for a user."""
        # Get interaction statistics
        stats = await self.interaction_repository.get_interaction_counts_by_type(user_id)
        
        # Get recent interactions
        recent_interactions = await self.interaction_repository.get_recent_interactions(
            user_id, days=30, limit=100
        )
        
        # Calculate behavioral features
        total_interactions = sum(stats.values())
        like_rate = stats.get(InteractionType.LIKE, 0) / max(total_interactions, 1)
        
        # Activity patterns
        recent_activity = len(recent_interactions)
        activity_streak = self._calculate_activity_streak(recent_interactions)
        
        return {
            "total_interactions": total_interactions,
            "like_rate": like_rate,
            "recent_activity_30d": recent_activity,
            "activity_streak_days": activity_streak,
            "interaction_diversity": len(stats)
        }
    
    def _calculate_activity_streak(self, interactions: List[Interaction]) -> int:
        """Calculate consecutive days of activity."""
        if not interactions:
            return 0
        
        dates = set(interaction.timestamp.date() for interaction in interactions)
        today = datetime.utcnow().date()
        streak = 0
        
        for i in range(30):  # Max 30 days lookback
            date = today - timedelta(days=i)
            if date in dates:
                streak += 1
            else:
                break
        
        return streak
    
    def get_feature_schema(self) -> Dict[str, str]:
        return {
            "total_interactions": "numeric",
            "like_rate": "numeric",
            "recent_activity_30d": "numeric",
            "activity_streak_days": "numeric",
            "interaction_diversity": "numeric"
        }


class CollaborativeFeatureExtractor(FeatureExtractor):
    """Extractor for collaborative filtering features."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def extract(self, user_id: int, **context: Any) -> Dict[str, Any]:
        """Extract collaborative filtering features."""
        # Get user embedding if available
        embedding_result = await self.session.execute(
            select(UserEmbedding)
            .where(UserEmbedding.user_id == user_id)
            .order_by(desc(UserEmbedding.created_at))
            .limit(1)
        )
        
        embedding = embedding_result.scalar_one_or_none()
        
        if not embedding:
            return {"has_embedding": False, "embedding_norm": 0.0}
        
        embedding_vector = np.array(embedding.embedding_vector)
        
        return {
            "has_embedding": True,
            "embedding_norm": float(np.linalg.norm(embedding_vector)),
            "embedding_sparsity": float(np.sum(embedding_vector == 0) / len(embedding_vector)),
            "model_version": embedding.model_version
        }
    
    def get_feature_schema(self) -> Dict[str, str]:
        return {
            "has_embedding": "boolean",
            "embedding_norm": "numeric",
            "embedding_sparsity": "numeric",
            "model_version": "categorical"
        }


class FeatureService:
    """
    Centralized service for feature management.
    
    Design Rationale:
    - Singleton pattern for feature consistency
    - Caching layer for performance
    - Feature versioning for reproducibility
    - Real-time and batch feature support
    """
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.user_repository = UserRepository(session)
        self.interaction_repository = InteractionRepository(session)
        
        # Initialize feature extractors
        self.extractors = {
            "demographic": DemographicFeatureExtractor(self.user_repository),
            "behavioral": BehavioralFeatureExtractor(self.interaction_repository),
            "collaborative": CollaborativeFeatureExtractor(session)
        }
        
        # Feature cache (in production, use Redis)
        self._feature_cache: Dict[str, Tuple[Dict[str, Any], datetime]] = {}
        self._cache_ttl = timedelta(minutes=5)
    
    async def get_features(
        self,
        user_id: int,
        feature_types: Optional[List[str]] = None,
        use_cache: bool = True,
        **context: Any
    ) -> Dict[str, Any]:
        """
        Get features for a user.
        
        Args:
            user_id: ID of the user
            feature_types: List of feature types to extract (default: all)
            use_cache: Whether to use cached features
            **context: Additional context for feature extraction
            
        Returns:
            Dictionary of features
        """
        cache_key = f"features:{user_id}:{str(sorted(feature_types or []))}"
        
        # Check cache
        if use_cache and cache_key in self._feature_cache:
            features, timestamp = self._feature_cache[cache_key]
            if datetime.utcnow() - timestamp < self._cache_ttl:
                logger.debug("Features retrieved from cache", user_id=user_id)
                return features
        
        # Extract features
        features = {}
        feature_types = feature_types or list(self.extractors.keys())
        
        for feature_type in feature_types:
            if feature_type in self.extractors:
                try:
                    extractor_features = await self.extractors[feature_type].extract(
                        user_id, **context
                    )
                    features.update(extractor_features)
                except Exception as e:
                    logger.error(
                        "Error extracting features",
                        user_id=user_id,
                        feature_type=feature_type,
                        error=str(e)
                    )
                    # Continue with other feature types
        
        # Cache features
        if use_cache:
            self._feature_cache[cache_key] = (features, datetime.utcnow())
        
        logger.debug("Features extracted", user_id=user_id, feature_count=len(features))
        return features
    
    async def get_feature_vector(
        self,
        user_id: int,
        feature_schema: Dict[str, str],
        **context: Any
    ) -> List[float]:
        """
        Get features as a numerical vector for ML models.
        
        Args:
            user_id: ID of the user
            feature_schema: Dictionary mapping feature names to types
            **context: Additional context for feature extraction
            
        Returns:
            List of numerical feature values
        """
        features = await self.get_features(user_id, **context)
        
        vector = []
        for feature_name, feature_type in feature_schema.items():
            value = features.get(feature_name, 0)
            
            if feature_type == "numeric":
                vector.append(float(value))
            elif feature_type == "boolean":
                vector.append(1.0 if value else 0.0)
            elif feature_type == "categorical":
                # For categorical features, we'll use a simple hash encoding
                # In production, use proper one-hot or embedding encoding
                vector.append(float(hash(str(value)) % 1000) / 1000.0)
            else:
                vector.append(0.0)  # Unknown type
        
        return vector
    
    async def store_features(
        self,
        user_id: int,
        features: Dict[str, Any],
        version: str = "v1"
    ) -> None:
        """
        Store features in the feature store.
        
        Args:
            user_id: ID of the user
            features: Dictionary of features to store
            version: Feature schema version
        """
        try:
            user_features = UserFeatures(
                user_id=user_id,
                feature_set=features,
                version=version
            )
            
            self.session.add(user_features)
            await self.session.commit()
            
            logger.info("Features stored", user_id=user_id, version=version)
            
        except Exception as e:
            await self.session.rollback()
            logger.error(
                "Error storing features",
                user_id=user_id,
                version=version,
                error=str(e)
            )
            raise
    
    async def get_stored_features(
        self,
        user_id: int,
        version: str = "v1",
        max_age_hours: int = 24
    ) -> Optional[Dict[str, Any]]:
        """
        Get features from the feature store.
        
        Args:
            user_id: ID of the user
            version: Feature schema version
            max_age_hours: Maximum age of features in hours
            
        Returns:
            Dictionary of features if found and not stale
        """
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
            
            query = (
                select(UserFeatures)
                .where(
                    and_(
                        UserFeatures.user_id == user_id,
                        UserFeatures.version == version,
                        UserFeatures.computed_at >= cutoff_time
                    )
                )
                .order_by(desc(UserFeatures.computed_at))
                .limit(1)
            )
            
            result = await self.session.execute(query)
            user_features = result.scalar_one_or_none()
            
            if user_features:
                logger.debug(
                    "Stored features retrieved",
                    user_id=user_id,
                    version=version,
                    age_hours=(datetime.utcnow() - user_features.computed_at).total_seconds() / 3600
                )
                return user_features.feature_set
            
            return None
            
        except Exception as e:
            logger.error(
                "Error retrieving stored features",
                user_id=user_id,
                version=version,
                error=str(e)
            )
            return None
    
    def get_feature_schema(self) -> Dict[str, str]:
        """Get the complete feature schema."""
        schema = {}
        for extractor in self.extractors.values():
            schema.update(extractor.get_feature_schema())
        return schema
    
    async def compute_and_store_features(
        self,
        user_id: int,
        version: str = "v1",
        **context: Any
    ) -> Dict[str, Any]:
        """
        Compute features and store them in the feature store.
        
        Args:
            user_id: ID of the user
            version: Feature schema version
            **context: Additional context for feature extraction
            
        Returns:
            Computed features
        """
        features = await self.get_features(user_id, use_cache=False, **context)
        await self.store_features(user_id, features, version)
        return features
    