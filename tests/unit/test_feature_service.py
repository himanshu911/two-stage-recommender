"""
Unit tests for FeatureService.

This module tests the FeatureService class with mocked dependencies.
Design Rationale:
- Mock external dependencies (repositories, database)
- Test feature extraction logic
- Verify caching behavior
- Test error handling and edge cases
"""

from typing import Dict, Any
from unittest.mock import Mock, AsyncMock, patch
import pytest
from datetime import datetime, timedelta

from app.services.feature_service import (
    FeatureService,
    DemographicFeatureExtractor,
    BehavioralFeatureExtractor,
    CollaborativeFeatureExtractor
)
from app.models.sql_models import User, Interaction
from app.core.logging import get_logger

logger = get_logger(__name__)


class TestFeatureService:
    """Test suite for FeatureService."""
    
    @pytest.fixture
    def mock_session(self) -> Mock:
        """Create a mock database session."""
        return Mock()
    
    @pytest.fixture
    def mock_user_repository(self) -> Mock:
        """Create a mock user repository."""
        return Mock()
    
    @pytest.fixture
    def mock_interaction_repository(self) -> Mock:
        """Create a mock interaction repository."""
        return Mock()
    
    @pytest.fixture
    def feature_service(self, mock_session: Mock) -> FeatureService:
        """Create FeatureService with mocked session."""
        return FeatureService(mock_session)
    
    @pytest.fixture
    def sample_user(self) -> User:
        """Create a sample user for testing."""
        return User(
            id=1,
            name="Test User",
            age=25,
            gender="female",
            location="San Francisco",
            bio="Test bio",
            interests=["hiking", "photography", "travel"],
            created_at=datetime.utcnow() - timedelta(days=30),
            last_active_at=datetime.utcnow() - timedelta(days=1)
        )
    
    @pytest.fixture
    def sample_interactions(self) -> list:
        """Create sample interactions for testing."""
        return [
            Interaction(
                id=1,
                user_id=1,
                target_user_id=2,
                interaction_type="like",
                timestamp=datetime.utcnow() - timedelta(days=5)
            ),
            Interaction(
                id=2,
                user_id=1,
                target_user_id=3,
                interaction_type="dislike",
                timestamp=datetime.utcnow() - timedelta(days=3)
            ),
            Interaction(
                id=3,
                user_id=1,
                target_user_id=4,
                interaction_type="super_like",
                timestamp=datetime.utcnow() - timedelta(days=1)
            )
        ]
    
    @pytest.mark.asyncio
    async def test_demographic_feature_extractor(self, sample_user: User):
        """Test demographic feature extraction."""
        # Create mock user repository
        mock_user_repo = Mock()
        mock_user_repo.get_by_id = AsyncMock(return_value=sample_user)
        
        # Create extractor
        extractor = DemographicFeatureExtractor(mock_user_repo)
        
        # Extract features
        features = await extractor.extract(1)
        
        # Verify features
        assert features["age"] == 25
        assert features["gender"] == "female"
        assert features["location"] == "San Francisco"
        assert features["interests_count"] == 3
        assert features["account_age_days"] == 30
    
    @pytest.mark.asyncio
    async def test_behavioral_feature_extractor(self, sample_interactions: list):
        """Test behavioral feature extraction."""
        # Create mock interaction repository
        mock_interaction_repo = Mock()
        mock_interaction_repo.get_interaction_counts_by_type = AsyncMock(return_value={
            "like": 20,
            "dislike": 10,
            "super_like": 5
        })
        mock_interaction_repo.get_recent_interactions = AsyncMock(return_value=sample_interactions)
        
        # Create extractor
        extractor = BehavioralFeatureExtractor(mock_interaction_repo)
        
        # Extract features
        features = await extractor.extract(1)
        
        # Verify features
        assert features["total_interactions"] == 35
        assert features["like_rate"] == 20/35
        assert features["recent_activity_30d"] == 3
        assert features["interaction_diversity"] == 3
    
    @pytest.mark.asyncio
    async def test_collaborative_feature_extractor_with_embedding(self):
        """Test collaborative feature extraction with user embedding."""
        # Create mock session and embedding
        mock_session = Mock()
        mock_embedding = Mock()
        mock_embedding.embedding_vector = [0.1, 0.2, 0.3, 0.4] * 16  # 64 dimensions
        mock_embedding.model_version = "v1.0"
        
        # Mock the query
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = mock_embedding
        mock_session.execute.return_value = mock_result
        
        # Create extractor
        extractor = CollaborativeFeatureExtractor(mock_session)
        
        # Extract features
        features = await extractor.extract(1)
        
        # Verify features
        assert features["has_embedding"] is True
        assert features["embedding_norm"] > 0
        assert features["model_version"] == "v1.0"
        assert 0 <= features["embedding_sparsity"] <= 1
    
    @pytest.mark.asyncio
    async def test_collaborative_feature_extractor_without_embedding(self):
        """Test collaborative feature extraction without user embedding."""
        # Create mock session with no embedding
        mock_session = Mock()
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result
        
        # Create extractor
        extractor = CollaborativeFeatureExtractor(mock_session)
        
        # Extract features
        features = await extractor.extract(1)
        
        # Verify features
        assert features["has_embedding"] is False
        assert features["embedding_norm"] == 0.0
    
    @pytest.mark.asyncio
    async def test_get_features_with_caching(self, feature_service: FeatureService):
        """Test feature retrieval with caching."""
        # Mock the extractors
        with patch.object(feature_service, 'extractors', {
            'demographic': Mock(),
            'behavioral': Mock()
        }):
            mock_demo_extractor = feature_service.extractors['demographic']
            mock_demo_extractor.extract = AsyncMock(return_value={"age": 25})
            mock_demo_extractor.get_feature_schema.return_value = {"age": "numeric"}
            
            mock_behavior_extractor = feature_service.extractors['behavioral']
            mock_behavior_extractor.extract = AsyncMock(return_value={"total_interactions": 50})
            mock_behavior_extractor.get_feature_schema.return_value = {"total_interactions": "numeric"}
            
            # First call - should extract features
            features1 = await feature_service.get_features(1, use_cache=True)
            
            # Second call - should use cache
            features2 = await feature_service.get_features(1, use_cache=True)
            
            # Verify features
            assert features1["age"] == 25
            assert features1["total_interactions"] == 50
            assert features2 == features1
            
            # Verify extractors were called only once
            mock_demo_extractor.extract.assert_called_once()
            mock_behavior_extractor.extract.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_features_without_caching(self, feature_service: FeatureService):
        """Test feature retrieval without caching."""
        # Mock the extractors
        with patch.object(feature_service, 'extractors', {
            'demographic': Mock()
        }):
            mock_extractor = feature_service.extractors['demographic']
            mock_extractor.extract = AsyncMock(return_value={"age": 25})
            mock_extractor.get_feature_schema.return_value = {"age": "numeric"}
            
            # Multiple calls without caching
            features1 = await feature_service.get_features(1, use_cache=False)
            features2 = await feature_service.get_features(1, use_cache=False)
            
            # Verify features are the same
            assert features1 == features2
            
            # Verify extractor was called twice
            assert mock_extractor.extract.call_count == 2
    
    @pytest.mark.asyncio
    async def test_get_feature_vector(self, feature_service: FeatureService):
        """Test feature vector conversion."""
        # Mock features
        features = {
            "age": 25,
            "gender": "female",
            "total_interactions": 50,
            "is_active": True
        }
        
        # Mock the get_features method
        feature_service.get_features = AsyncMock(return_value=features)
        
        # Define feature schema
        schema = {
            "age": "numeric",
            "gender": "categorical",
            "total_interactions": "numeric",
            "is_active": "boolean"
        }
        
        # Get feature vector
        vector = await feature_service.get_feature_vector(1, schema)
        
        # Verify vector
        assert len(vector) == 4
        assert vector[0] == 25.0  # age
        assert 0 <= vector[1] <= 1.0  # gender (hashed)
        assert vector[2] == 50.0  # total_interactions
        assert vector[3] == 1.0  # is_active
    
    @pytest.mark.asyncio
    async def test_store_and_retrieve_features(self, feature_service: FeatureService):
        """Test storing and retrieving features from feature store."""
        # Mock features
        features = {"age": 25, "total_interactions": 50}
        
        # Mock database operations
        feature_service.session.add = Mock()
        feature_service.session.commit = AsyncMock()
        
        # Store features
        await feature_service.store_features(1, features, version="v1")
        
        # Verify database operations
        feature_service.session.add.assert_called_once()
        feature_service.session.commit.assert_called_once()
        
        # Mock retrieval
        mock_user_features = Mock()
        mock_user_features.feature_set = features
        mock_user_features.computed_at = datetime.utcnow() - timedelta(hours=1)
        
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = mock_user_features
        feature_service.session.execute.return_value = mock_result
        
        # Retrieve features
        retrieved_features = await feature_service.get_stored_features(1, version="v1", max_age_hours=2)
        
        # Verify retrieved features
        assert retrieved_features == features
    
    @pytest.mark.asyncio
    async def test_get_stored_features_expired(self, feature_service: FeatureService):
        """Test retrieving expired features from feature store."""
        # Mock expired features
        mock_user_features = Mock()
        mock_user_features.computed_at = datetime.utcnow() - timedelta(hours=5)
        
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = mock_user_features
        feature_service.session.execute.return_value = mock_result
        
        # Retrieve features with 2-hour max age
        retrieved_features = await feature_service.get_stored_features(1, max_age_hours=2)
        
        # Verify no features returned (expired)
        assert retrieved_features is None
    
    @pytest.mark.asyncio
    async def test_error_handling_in_extraction(self, feature_service: FeatureService):
        """Test error handling during feature extraction."""
        # Mock a failing extractor
        with patch.object(feature_service, 'extractors', {
            'demographic': Mock(),
            'behavioral': Mock()
        }):
            mock_demo_extractor = feature_service.extractors['demographic']
            mock_demo_extractor.extract = AsyncMock(return_value={"age": 25})
            mock_demo_extractor.get_feature_schema.return_value = {"age": "numeric"}
            
            mock_behavior_extractor = feature_service.extractors['behavioral']
            mock_behavior_extractor.extract = AsyncMock(side_effect=Exception("DB error"))
            mock_behavior_extractor.get_feature_schema.return_value = {"total_interactions": "numeric"}
            
            # Extract features (should handle the error gracefully)
            features = await feature_service.get_features(1, feature_types=["demographic", "behavioral"])
            
            # Verify that demographic features were still extracted
            assert "age" in features
            assert features["age"] == 25
            
            # Verify that behavioral features are not present due to error
            assert "total_interactions" not in features
    
    def test_get_feature_schema(self, feature_service: FeatureService):
        """Test getting complete feature schema."""
        # Mock extractors with different schemas
        with patch.object(feature_service, 'extractors', {
            'demo': Mock(),
            'behavior': Mock()
        }):
            feature_service.extractors['demo'].get_feature_schema.return_value = {"age": "numeric"}
            feature_service.extractors['behavior'].get_feature_schema.return_value = {"activity": "numeric"}
            
            # Get complete schema
            schema = feature_service.get_feature_schema()
            
            # Verify schema
            assert schema["age"] == "numeric"
            assert schema["activity"] == "numeric"
            assert len(schema) == 2
            