"""
Ranking Service for recommendation pipeline.

This service implements the second stage of the two-stage recommendation system.
Design Rationale:
- Logistic regression model for ranking
- Real-time feature computation
- Probability-based scoring
- Model versioning and A/B testing support
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import pickle
import numpy as np
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import User, MLModel
from app.services.feature_service import FeatureService
from app.repositories.user_repository import UserRepository
from app.core.logging import get_logger
from app.core.config import get_settings

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class RankedCandidate:
    """Represents a ranked candidate with probability score."""
    user_id: int
    score: float  # Probability of positive interaction
    features: Dict[str, Any]
    model_version: str


class LogisticRegressionModel:
    """
    Logistic regression model for ranking candidates.
    
    Design Rationale:
    - Simple, interpretable model for ranking
    - Supports probability-based scoring
    - Feature scaling for better convergence
    - Model persistence for production serving
    """
    
    def __init__(self, regularization_strength: float = 1.0):
        self.model = LogisticRegression(
            C=regularization_strength,
            random_state=42,
            max_iter=1000
        )
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.is_trained = False
        
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, float]:
        """
        Train the logistic regression model.
        
        Args:
            X: Feature matrix
            y: Target labels
            feature_names: Names of features
            
        Returns:
            Dictionary with training metrics
        """
        try:
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model.fit(X_scaled, y)
            self.feature_names = feature_names
            self.is_trained = True
            
            # Calculate metrics
            train_accuracy = self.model.score(X_scaled, y)
            
            metrics = {
                "train_accuracy": float(train_accuracy),
                "n_features": len(feature_names),
                "n_samples": len(X),
                "feature_importance": dict(zip(feature_names, abs(self.model.coef_[0])))
            }
            
            logger.info("Ranking model trained", metrics=metrics)
            return metrics
            
        except Exception as e:
            logger.error("Error training ranking model", error=str(e))
            raise
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities for input features."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]  # Probability of positive class
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the model."""
        if not self.is_trained or not self.feature_names:
            return {}
        
        return dict(zip(self.feature_names, abs(self.model.coef_[0])))


class RankingService:
    """
    Service for ranking candidate recommendations.
    
    Design Rationale:
    - Second stage of two-stage recommendation
    - Real-time feature computation
    - Model versioning for A/B testing
    - Feature importance tracking
    """
    
    def __init__(
        self,
        session: AsyncSession,
        feature_service: FeatureService,
        model_version: str = "v1"
    ):
        self.session = session
        self.feature_service = feature_service
        self.model_version = model_version
        
        # Initialize model
        self.model = LogisticRegressionModel()
        
        # Feature schema for ranking model
        self.feature_schema = [
            "age",
            "account_age_days",
            "interests_count",
            "total_interactions",
            "like_rate",
            "recent_activity_30d",
            "activity_streak_days",
            "embedding_norm",
            "embedding_sparsity"
        ]
        
        self.is_model_loaded = False
    
    async def rank_candidates(
        self,
        user_id: int,
        candidate_ids: List[int],
        limit: int = 20
    ) -> List[RankedCandidate]:
        """
        Rank candidate users for recommendation.
        
        Args:
            user_id: ID of the target user
            candidate_ids: List of candidate user IDs
            limit: Maximum number of candidates to return
            
        Returns:
            List of ranked candidates with scores
        """
        try:
            # Ensure model is loaded
            if not self.is_model_loaded:
                await self._load_model()
            
            # If model not available, use fallback ranking
            if not self.model.is_trained:
                return await self._fallback_ranking(user_id, candidate_ids, limit)
            
            # Get features for all candidates
            candidate_features = []
            valid_candidate_ids = []
            
            for candidate_id in candidate_ids:
                features = await self._get_candidate_features(user_id, candidate_id)
                if features:
                    feature_vector = self._features_to_vector(features)
                    candidate_features.append(feature_vector)
                    valid_candidate_ids.append(candidate_id)
            
            if not candidate_features:
                return []
            
            # Predict probabilities
            feature_matrix = np.array(candidate_features)
            probabilities = self.model.predict_proba(feature_matrix)
            
            # Create ranked candidates
            ranked_candidates = []
            for candidate_id, probability, features in zip(
                valid_candidate_ids, probabilities, candidate_features
            ):
                feature_dict = dict(zip(self.feature_schema, features))
                ranked_candidates.append(RankedCandidate(
                    user_id=candidate_id,
                    score=float(probability),
                    features=feature_dict,
                    model_version=self.model_version
                ))
            
            # Sort by score and apply limit
            ranked_candidates.sort(key=lambda x: x.score, reverse=True)
            
            logger.info(
                "Candidates ranked",
                user_id=user_id,
                candidate_count=len(ranked_candidates),
                top_score=ranked_candidates[0].score if ranked_candidates else 0
            )
            
            return ranked_candidates[:limit]
            
        except Exception as e:
            logger.error("Error ranking candidates", user_id=user_id, error=str(e))
            return []
    
    async def _get_candidate_features(
        self,
        user_id: int,
        candidate_id: int
    ) -> Optional[Dict[str, Any]]:
        """Get features for a candidate user."""
        try:
            # Get user features
            user_features = await self.feature_service.get_features(user_id)
            
            # Get candidate features
            candidate_features = await self.feature_service.get_features(candidate_id)
            
            # Combine features
            features = {}
            
            # Candidate features (primary)
            for feature in self.feature_schema:
                if feature in candidate_features:
                    features[feature] = candidate_features[feature]
                else:
                    features[feature] = 0.0
            
            # Cross-features (user-candidate interactions)
            # For now, using simple features - in production, add more sophisticated cross-features
            
            return features
            
        except Exception as e:
            logger.error(
                "Error getting candidate features",
                user_id=user_id,
                candidate_id=candidate_id,
                error=str(e)
            )
            return None
    
    def _features_to_vector(self, features: Dict[str, Any]) -> List[float]:
        """Convert feature dictionary to vector."""
        vector = []
        for feature_name in self.feature_schema:
            value = features.get(feature_name, 0)
            if isinstance(value, (int, float)):
                vector.append(float(value))
            elif isinstance(value, bool):
                vector.append(1.0 if value else 0.0)
            else:
                vector.append(0.0)  # Unknown type
        return vector
    
    async def _load_model(self) -> None:
        """Load the ranking model from database."""
        try:
            # Try to load from database
            model_record = await self.session.execute(
                select(MLModel)
                .where(
                    MLModel.model_type == "ranking",
                    MLModel.version == self.model_version,
                    MLModel.is_active == True
                )
                .order_by(MLModel.created_at.desc())
                .limit(1)
            )
            
            model_data = model_record.scalar_one_or_none()
            
            if model_data:
                # Load model from binary data
                model_binary = model_data.model_binary
                
                # Deserialize the model
                self.model = pickle.loads(model_binary)
                
                logger.info("Model loaded from database", version=self.model_version)
            
            self.is_model_loaded = True
            
        except Exception as e:
            logger.error("Error loading model", error=str(e))
            self.is_model_loaded = True  # Don't try again
    
    async def train_and_save_model(
        self,
        training_data: List[Tuple[Dict[str, Any], bool]],
        feature_schema: List[str]
    ) -> Dict[str, Any]:
        """
        Train and save the ranking model.
        
        Args:
            training_data: List of (features, label) tuples
            feature_schema: List of feature names
            
        Returns:
            Training metrics
        """
        try:
            # Prepare training data
            X = []
            y = []
            
            for features, label in training_data:
                feature_vector = []
                for feature_name in feature_schema:
                    feature_vector.append(features.get(feature_name, 0))
                X.append(feature_vector)
                y.append(label)
            
            X = np.array(X)
            y = np.array(y)
            
            # Train model
            metrics = self.model.train(X, y, feature_schema)
            
            # Save model to database
            model_binary = pickle.dumps(self.model)
            
            ml_model = MLModel(
                model_type="ranking",
                version=self.model_version,
                model_binary=model_binary,
                metrics=metrics,
                is_active=True
            )
            
            self.session.add(ml_model)
            await self.session.commit()
            
            logger.info("Model trained and saved", version=self.model_version, metrics=metrics)
            
            return metrics
            
        except Exception as e:
            await self.session.rollback()
            logger.error("Error training and saving model", error=str(e))
            raise
    
    async def _fallback_ranking(
        self,
        user_id: int,
        candidate_ids: List[int],
        limit: int
    ) -> List[RankedCandidate]:
        """Fallback ranking when ML model is not available."""
        try:
            # Simple heuristic-based ranking
            # Score based on: account age, recent activity, profile completeness
            
            ranked_candidates = []
            
            for candidate_id in candidate_ids:
                # Get basic features
                features = await self.feature_service.get_features(candidate_id)
                
                # Simple heuristic score
                score = 0.0
                
                # Account age (newer accounts get slight boost)
                account_age = features.get("account_age_days", 0)
                if 1 <= account_age <= 30:  # Recently joined
                    score += 0.3
                
                # Activity level
                recent_activity = features.get("recent_activity_30d", 0)
                if recent_activity > 0:
                    score += 0.4
                
                # Profile completeness
                if features.get("bio"):
                    score += 0.2
                if len(features.get("interests", [])) >= 3:
                    score += 0.1
                
                ranked_candidates.append(RankedCandidate(
                    user_id=candidate_id,
                    score=score,
                    features=features,
                    model_version="fallback"
                ))
            
            # Sort by score
            ranked_candidates.sort(key=lambda x: x.score, reverse=True)
            
            logger.debug("Fallback ranking completed", user_id=user_id, count=len(ranked_candidates))
            
            return ranked_candidates[:limit]
            
        except Exception as e:
            logger.error("Error in fallback ranking", user_id=user_id, error=str(e))
            return []
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the current model."""
        return self.model.get_feature_importance()
    
    async def update_model_version(self, version: str) -> None:
        """Update the model version and reload."""
        self.model_version = version
        self.is_model_loaded = False
        await self._load_model()
        