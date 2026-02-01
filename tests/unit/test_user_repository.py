"""
Unit tests for UserRepository.

This module tests the UserRepository class with mocked database sessions.
Design Rationale:
- Mock database sessions for isolated testing
- Test both happy paths and error scenarios
- Use pytest fixtures for test data
- Comprehensive coverage of repository methods
"""

from typing import List
from unittest.mock import Mock, AsyncMock, patch
import pytest
from datetime import datetime, timedelta

from app.models.sql_models import User
from app.repositories.user_repository import UserRepository
from app.core.logging import get_logger

logger = get_logger(__name__)


class TestUserRepository:
    """Test suite for UserRepository."""
    
    @pytest.fixture
    def mock_session(self) -> Mock:
        """Create a mock database session."""
        session = Mock()
        session.execute = AsyncMock()
        session.commit = AsyncMock()
        session.rollback = AsyncMock()
        session.close = AsyncMock()
        return session
    
    @pytest.fixture
    def user_repository(self, mock_session: Mock) -> UserRepository:
        """Create UserRepository with mocked session."""
        return UserRepository(mock_session)
    
    @pytest.fixture
    def sample_users(self) -> List[User]:
        """Create sample users for testing."""
        return [
            User(
                id=1,
                name="Alice",
                age=25,
                gender="female",
                location="San Francisco",
                bio="Love hiking",
                interests=["hiking", "photography"],
                created_at=datetime.utcnow()
            ),
            User(
                id=2,
                name="Bob",
                age=30,
                gender="male",
                location="New York",
                bio="Software engineer",
                interests=["technology", "cooking"],
                created_at=datetime.utcnow()
            ),
            User(
                id=3,
                name="Charlie",
                age=28,
                gender="male",
                location="San Francisco",
                bio="Artist and musician",
                interests=["art", "music"],
                created_at=datetime.utcnow()
            )
        ]
    
    @pytest.mark.asyncio
    async def test_get_by_location(self, user_repository: UserRepository, mock_session: Mock, sample_users: List[User]):
        """Test getting users by location."""
        # Mock the query result
        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = [sample_users[0], sample_users[2]]  # SF users
        mock_session.execute.return_value = mock_result
        
        # Execute the method
        users = await user_repository.get_by_location("San Francisco", skip=0, limit=10)
        
        # Verify the result
        assert len(users) == 2
        assert all(user.location == "San Francisco" for user in users)
        assert users[0].name in ["Alice", "Charlie"]
        
        # Verify the query was called correctly
        mock_session.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_by_age_range(self, user_repository: UserRepository, mock_session: Mock, sample_users: List[User]):
        """Test getting users by age range."""
        # Mock the query result
        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = [sample_users[0], sample_users[2]]  # Age 25-28
        mock_session.execute.return_value = mock_result
        
        # Execute the method
        users = await user_repository.get_by_age_range(25, 28, skip=0, limit=10)
        
        # Verify the result
        assert len(users) == 2
        assert all(25 <= user.age <= 28 for user in users)
        
        # Verify the query was called correctly
        mock_session.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_active_users(self, user_repository: UserRepository, mock_session: Mock, sample_users: List[User]):
        """Test getting active users."""
        # Set recent activity for some users
        sample_users[0].last_active_at = datetime.utcnow() - timedelta(days=10)
        sample_users[1].last_active_at = datetime.utcnow() - timedelta(days=40)  # Not active
        sample_users[2].last_active_at = datetime.utcnow() - timedelta(days=5)
        
        # Mock the query result
        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = [sample_users[0], sample_users[2]]
        mock_session.execute.return_value = mock_result
        
        # Execute the method
        users = await user_repository.get_active_users(days_since_active=30, skip=0, limit=10)
        
        # Verify the result
        assert len(users) == 2
        assert "Bob" not in [user.name for user in users]  # Bob is not active
        
    @pytest.mark.asyncio
    async def test_get_users_with_interest(self, user_repository: UserRepository, mock_session: Mock, sample_users: List[User]):
        """Test getting users with specific interest."""
        # Mock the query result
        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = [sample_users[2]]  # Only Charlie has "art"
        mock_session.execute.return_value = mock_result
        
        # Execute the method
        users = await user_repository.get_users_with_interest("art", skip=0, limit=10)
        
        # Verify the result
        assert len(users) == 1
        assert users[0].name == "Charlie"
        assert "art" in users[0].interests
    
    @pytest.mark.asyncio
    async def test_get_users_not_interacted_with(self, user_repository: UserRepository, mock_session: Mock):
        """Test getting users not interacted with."""
        # Mock the subquery and main query results
        mock_subquery_result = Mock()
        mock_subquery_result.scalars.return_value.all.return_value = [2, 3]  # Already interacted with
        
        mock_main_result = Mock()
        mock_main_result.scalars.return_value.all.return_value = [
            User(id=4, name="David", age=26, gender="male", location="SF", interests=["sports"])
        ]
        
        mock_session.execute.side_effect = [mock_main_result, mock_subquery_result]
        
        # Execute the method
        users = await user_repository.get_users_not_interacted_with(1, skip=0, limit=10)
        
        # Verify the result
        assert len(users) == 1
        assert users[0].id == 4  # Only David not interacted with
        assert users[0].name == "David"
    
    @pytest.mark.asyncio
    async def test_get_interaction_statistics(self, user_repository: UserRepository, mock_session: Mock):
        """Test getting interaction statistics."""
        # Mock the query result
        mock_result = Mock()
        mock_result.all.return_value = [
            ("like", 25),
            ("dislike", 10),
            ("super_like", 5)
        ]
        mock_session.execute.return_value = mock_result
        
        # Execute the method
        stats = await user_repository.get_interaction_statistics(1)
        
        # Verify the result
        assert stats == {"like": 25, "dislike": 10, "super_like": 5}
        assert stats["like"] == 25
        assert sum(stats.values()) == 40
    
    @pytest.mark.asyncio
    async def test_update_last_active(self, user_repository: UserRepository, mock_session: Mock):
        """Test updating user's last active timestamp."""
        # Mock the update result
        mock_result = Mock()
        mock_result.rowcount = 1  # Successfully updated 1 row
        mock_session.execute.return_value = mock_result
        
        # Execute the method
        success = await user_repository.update_last_active(1)
        
        # Verify the result
        assert success is True
        mock_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_error_handling(self, user_repository: UserRepository, mock_session: Mock):
        """Test error handling in repository methods."""
        # Mock an exception during database execution
        mock_session.execute.side_effect = Exception("Database connection failed")
        
        # Execute the method and expect an exception
        with pytest.raises(Exception) as exc_info:
            await user_repository.get_by_location("San Francisco")
        
        # Verify error handling
        assert "Database connection failed" in str(exc_info.value)
        mock_session.rollback.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_empty_results(self, user_repository: UserRepository, mock_session: Mock):
        """Test handling of empty query results."""
        # Mock empty result
        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute.return_value = mock_result
        
        # Execute the method
        users = await user_repository.get_by_location("Nonexistent City")
        
        # Verify empty result handling
        assert len(users) == 0
        assert isinstance(users, list)
        