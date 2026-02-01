"""
Candidate Generation Service for recommendation pipeline.

This service implements the first stage of the two-stage recommendation system.
Design Rationale:
- Collaborative filtering using matrix factorization
- Efficient similarity search using FAISS
- Real-time candidate generation
- Scalable architecture for large user bases
"""

from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime
import numpy as np
from dataclasses import dataclass

import faiss
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_

from app.models.database import User, UserEmbedding, Interaction
from app.repositories.user_repository import UserRepository
from app.repositories.interaction_repository import InteractionRepository
from app.services.feature_service import FeatureService
from app.core.logging import get_logger
from app.core.config import get_settings

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class Candidate:
    """Represents a candidate user for recommendation."""
    user_id: int
    score: float
    source: str  # e.g., "collaborative_filtering", "content_based", "random"
    metadata: Dict[str, Any]


class CollaborativeFilteringModel:
    """
    Matrix factorization model for collaborative filtering.
    
    Design Rationale:
    - Uses matrix factorization for user embeddings
    - FAISS for efficient similarity search
    - Supports online and offline training
    - Configurable embedding dimensions
    """
    
    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim
        self.user_embeddings: Dict[int, np.ndarray] = {}
        self.user_index: Optional[faiss.Index] = None
        self.item_embeddings: Dict[int, np.ndarray] = {}
        self.is_trained = False
        
    def train_offline(
        self,
        user_item_matrix: np.ndarray,
        user_ids: List[int],
        item_ids: List[int],
        epochs: int = 100,
        learning_rate: float = 0.01,
        reg_coefficient: float = 0.01
    ) -> None:
        """
        Train the model offline using matrix factorization.
        
        Args:
            user_item_matrix: User-item interaction matrix
            user_ids: List of user IDs corresponding to matrix rows
            item_ids: List of item IDs corresponding to matrix columns
            epochs: Number of training epochs
            learning_rate: Learning rate for SGD
            reg_coefficient: Regularization coefficient
        """
        try:
            n_users, n_items = user_item_matrix.shape
            
            # Initialize embeddings
            self.user_embeddings = {
                user_id: np.random.normal(0, 0.1, self.embedding_dim)
                for user_id in user_ids
            }
            
            self.item_embeddings = {
                item_id: np.random.normal(0, 0.1, self.embedding_dim)
                for item_id in item_ids
            }
            
            # Training loop (simplified SGD)
            for epoch in range(epochs):
                total_loss = 0
                
                for i, user_id in enumerate(user_ids):
                    for j, item_id in enumerate(item_ids):
                        if user_item_matrix[i, j] > 0:  # Only train on observed interactions
                            # Predict
                            user_emb = self.user_embeddings[user_id]
                            item_emb = self.item_embeddings[item_id]
                            prediction = np.dot(user_emb, item_emb)
                            
                            # Calculate error
                            error = user_item_matrix[i, j] - prediction
                            total_loss += error ** 2
                            
                            # Update embeddings
                            self.user_embeddings[user_id] += learning_rate * (
                                error * item_emb - reg_coefficient * user_emb
                            )
                            self.item_embeddings[item_id] += learning_rate * (
                                error * user_emb - reg_coefficient * item_emb
                            )
                
                if epoch % 10 == 0:
                    logger.debug(
                        "Training epoch completed",
                        epoch=epoch,
                        loss=total_loss
                    )
            
            # Build FAISS index for efficient similarity search
            self._build_index()
            self.is_trained = True
            
            logger.info("Collaborative filtering model trained", epochs=epochs)
            
        except Exception as e:
            logger.error("Error training collaborative filtering model", error=str(e))
            raise
    
    def _build_index(self) -> None:
        """Build FAISS index for efficient similarity search."""
        try:
            if not self.user_embeddings:
                return
            
            # Convert embeddings to matrix
            user_ids = list(self.user_embeddings.keys())
            embeddings_matrix = np.array([
                self.user_embeddings[user_id] for user_id in user_ids
            ]).astype(np.float32)
            
            # Create FAISS index
            self.user_index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
            self.user_index.add(embeddings_matrix)
            
            # Store user ID mapping
            self.user_id_mapping = {i: user_id for i, user_id in enumerate(user_ids)}
            
            logger.info("FAISS index built", n_users=len(user_ids))
            
        except Exception as e:
            logger.error("Error building FAISS index", error=str(e))
            raise
    
    def find_similar_users(
        self,
        user_id: int,
        k: int = 100,
        exclude_users: Optional[List[int]] = None
    ) -> List[Tuple[int, float]]:
        """
        Find similar users based on embeddings.
        
        Args:
            user_id: ID of the target user
            k: Number of similar users to return
            exclude_users: List of user IDs to exclude
            
        Returns:
            List of (user_id, similarity_score) tuples
        """
        if not self.is_trained or user_id not in self.user_embeddings:
            return []
        
        try:
            # Get user embedding
            user_embedding = self.user_embeddings[user_id].astype(np.float32)
            
            # Search for similar users
            if self.user_index is not None:
                similarities, indices = self.user_index.search(
                    user_embedding.reshape(1, -1), k + 1
                )
                
                similar_users = []
                for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                    if idx >= 0 and self.user_id_mapping[idx] != user_id:
                        similar_user_id = self.user_id_mapping[idx]
                        
                        # Check exclusion list
                        if exclude_users and similar_user_id in exclude_users:
                            continue
                        
                        similar_users.append((similar_user_id, float(similarity)))
                
                return similar_users[:k]
            
            return []
            
        except Exception as e:
            logger.error("Error finding similar users", user_id=user_id, error=str(e))
            return []


class CandidateGenerationService:
    """
    Service for generating candidate recommendations.
    
    Design Rationale:
    - Implements two-stage recommendation pipeline
    - Multiple candidate generation strategies
    - Efficient filtering and deduplication
    - Configurable candidate pool size
    """
    
    def __init__(
        self,
        session: AsyncSession,
        feature_service: FeatureService,
        candidate_pool_size: int = 100
    ):
        self.session = session
        self.feature_service = feature_service
        self.candidate_pool_size = candidate_pool_size
        
        # Initialize repositories
        self.user_repository = UserRepository(session)
        self.interaction_repository = InteractionRepository(session)
        
        # Initialize collaborative filtering model
        self.cf_model = CollaborativeFilteringModel(
            embedding_dim=settings.EMBEDDING_DIMENSION
        )
        
        # Cache for user embeddings
        self._embedding_cache: Dict[int, np.ndarray] = {}
    
    async def generate_candidates(
        self,
        user_id: int,
        limit: int = 50,
        exclude_seen: bool = True,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Candidate]:
        """
        Generate candidate recommendations for a user.
        
        Args:
            user_id: ID of the target user
            limit: Maximum number of candidates to return
            exclude_seen: Whether to exclude users already seen
            filters: Additional filters (age_range, location, etc.)
            
        Returns:
            List of candidate users
        """
        try:
            candidates = []
            
            # Get users to exclude (already seen)
            exclude_users = []
            if exclude_seen:
                exclude_users = await self._get_seen_users(user_id)
            exclude_users.append(user_id)  # Exclude self
            
            # 1. Collaborative Filtering candidates
            cf_candidates = await self._generate_cf_candidates(
                user_id, limit // 2, exclude_users
            )
            candidates.extend(cf_candidates)
            
            # 2. Content-based candidates
            content_candidates = await self._generate_content_candidates(
                user_id, limit // 3, exclude_users, filters
            )
            candidates.extend(content_candidates)
            
            # 3. Random candidates for exploration
            random_candidates = await self._generate_random_candidates(
                user_id, limit - len(candidates), exclude_users, filters
            )
            candidates.extend(random_candidates)
            
            # 4. Deduplicate and rank
            candidates = self._deduplicate_candidates(candidates)
            candidates = self._rank_candidates(candidates)
            
            # 5. Apply final limit
            candidates = candidates[:limit]
            
            logger.info(
                "Candidates generated",
                user_id=user_id,
                candidate_count=len(candidates),
                sources=list(set(c.source for c in candidates))
            )
            
            return candidates
            
        except Exception as e:
            logger.error("Error generating candidates", user_id=user_id, error=str(e))
            return []
    
    async def _generate_cf_candidates(
        self,
        user_id: int,
        limit: int,
        exclude_users: List[int]
    ) -> List[Candidate]:
        """Generate collaborative filtering candidates."""
        try:
            # Ensure model is trained
            if not self.cf_model.is_trained:
                await self._train_cf_model()
            
            # Find similar users
            similar_users = self.cf_model.find_similar_users(
                user_id, k=limit * 2, exclude_users=exclude_users
            )
            
            candidates = []
            for similar_user_id, similarity_score in similar_users[:limit]:
                candidates.append(Candidate(
                    user_id=similar_user_id,
                    score=similarity_score,
                    source="collaborative_filtering",
                    metadata={"similarity": similarity_score}
                ))
            
            return candidates
            
        except Exception as e:
            logger.error("Error generating CF candidates", user_id=user_id, error=str(e))
            return []
    
    async def _generate_content_candidates(
        self,
        user_id: int,
        limit: int,
        exclude_users: List[int],
        filters: Optional[Dict[str, Any]]
    ) -> List[Candidate]:
        """Generate content-based candidates."""
        try:
            # Get target user features
            user_features = await self.feature_service.get_features(user_id)
            
            # Query potential candidates
            query = select(User).where(User.id.not_in(exclude_users))
            
            # Apply filters
            if filters:
                if "min_age" in filters:
                    query = query.where(User.age >= filters["min_age"])
                if "max_age" in filters:
                    query = query.where(User.age <= filters["max_age"])
                if "location" in filters:
                    query = query.where(User.location.ilike(f"%{filters['location']}%"))
            
            query = query.limit(limit * 2)
            
            result = await self.session.execute(query)
            potential_candidates = result.scalars().all()
            
            # Score candidates based on content similarity
            candidates = []
            for candidate in potential_candidates:
                # Simple content scoring (age difference, location, interests)
                score = self._calculate_content_score(user_features, candidate)
                
                if score > 0:  # Only include if there's some similarity
                    candidates.append(Candidate(
                        user_id=candidate.id,
                        score=score,
                        source="content_based",
                        metadata={
                            "age_diff": abs(user_features.get("age", 0) - candidate.age),
                            "common_interests": len(set(user_features.get("interests", [])) & set(candidate.interests or []))
                        }
                    ))
            
            # Sort by score and return top candidates
            candidates.sort(key=lambda x: x.score, reverse=True)
            return candidates[:limit]
            
        except Exception as e:
            logger.error("Error generating content candidates", user_id=user_id, error=str(e))
            return []
    
    async def _generate_random_candidates(
        self,
        user_id: int,
        limit: int,
        exclude_users: List[int],
        filters: Optional[Dict[str, Any]]
    ) -> List[Candidate]:
        """Generate random candidates for exploration."""
        try:
            query = select(User).where(User.id.not_in(exclude_users))
            
            # Apply filters if provided
            if filters:
                if "min_age" in filters:
                    query = query.where(User.age >= filters["min_age"])
                if "max_age" in filters:
                    query = query.where(User.age <= filters["max_age"])
            
            # Random ordering (database-specific implementation needed)
            query = query.order_by(func.random()).limit(limit)
            
            result = await self.session.execute(query)
            random_users = result.scalars().all()
            
            candidates = []
            for user in random_users:
                candidates.append(Candidate(
                    user_id=user.id,
                    score=0.1,  # Low base score for random candidates
                    source="random",
                    metadata={"exploration": True}
                ))
            
            return candidates
            
        except Exception as e:
            logger.error("Error generating random candidates", user_id=user_id, error=str(e))
            return []
    
    def _calculate_content_score(self, user_features: Dict[str, Any], candidate: User) -> float:
        """Calculate content-based similarity score."""
        score = 0.0
        
        # Age similarity (inverse of age difference, normalized)
        age_diff = abs(user_features.get("age", 0) - candidate.age)
        age_score = max(0, 1 - (age_diff / 10))  # Decay over 10 years
        score += age_score * 0.3
        
        # Location similarity
        if user_features.get("location") == candidate.location:
            score += 0.3
        
        # Interest similarity (Jaccard similarity)
        user_interests = set(user_features.get("interests", []))
        candidate_interests = set(candidate.interests or [])
        
        if user_interests or candidate_interests:
            intersection = len(user_interests & candidate_interests)
            union = len(user_interests | candidate_interests)
            interest_score = intersection / union if union > 0 else 0
            score += interest_score * 0.4
        
        return score
    
    async def _get_seen_users(self, user_id: int) -> List[int]:
        """Get list of users that the target user has already seen."""
        try:
            query = (
                select(Interaction.target_user_id)
                .where(Interaction.user_id == user_id)
                .distinct()
            )
            
            result = await self.session.execute(query)
            seen_users = result.scalars().all()
            
            return list(seen_users)
            
        except Exception as e:
            logger.error("Error getting seen users", user_id=user_id, error=str(e))
            return []
    
    def _deduplicate_candidates(self, candidates: List[Candidate]) -> List[Candidate]:
        """Remove duplicate candidates, keeping the highest score."""
        seen = {}
        for candidate in candidates:
            if candidate.user_id not in seen or candidate.score > seen[candidate.user_id].score:
                seen[candidate.user_id] = candidate
        
        return list(seen.values())
    
    def _rank_candidates(self, candidates: List[Candidate]) -> List[Candidate]:
        """Rank candidates by score."""
        return sorted(candidates, key=lambda x: x.score, reverse=True)
    
    async def _train_cf_model(self) -> None:
        """Train the collaborative filtering model."""
        try:
            # Get interaction data
            query = (
                select(Interaction.user_id, Interaction.target_user_id, Interaction.interaction_type)
                .where(Interaction.interaction_type == "like")
            )
            
            result = await self.session.execute(query)
            interactions = result.all()
            
            if len(interactions) < 100:  # Need minimum data for training
                logger.warning("Insufficient interaction data for CF training")
                return
            
            # Build user-item matrix
            users = list(set(i[0] for i in interactions))
            items = list(set(i[1] for i in interactions))
            
            user_to_idx = {user_id: idx for idx, user_id in enumerate(users)}
            item_to_idx = {item_id: idx for idx, item_id in enumerate(items)}
            
            user_item_matrix = np.zeros((len(users), len(items)))
            for user_id, item_id, interaction_type in interactions:
                user_idx = user_to_idx[user_id]
                item_idx = item_to_idx[item_id]
                
                # Weight different interaction types
                if interaction_type == "like":
                    weight = 1.0
                elif interaction_type == "super_like":
                    weight = 2.0
                else:
                    weight = 0.5
                
                user_item_matrix[user_idx, item_idx] = weight
            
            # Train model
            self.cf_model.train_offline(
                user_item_matrix,
                users,
                items,
                epochs=50
            )
            
            logger.info("CF model trained", n_interactions=len(interactions), n_users=len(users))
            
        except Exception as e:
            logger.error("Error training CF model", error=str(e))
            # Continue without CF model
            