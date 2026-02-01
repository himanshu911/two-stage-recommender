"""
Integration tests for User API endpoints.

This module tests the complete API flow from request to database.
Design Rationale:
- Test complete API flow with real database
- Use test database isolation
- Test both success and error scenarios
- Verify database state changes
"""

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.sql_models import User
from app.core.db import SQLModel
from app.models.schemas import UserCreateRequest, UserUpdateRequest


class TestUserAPI:
    """Integration tests for User API endpoints."""
    
    @pytest.mark.asyncio
    async def test_create_user(self, async_client: AsyncClient, db_session: AsyncSession):
        """Test creating a new user via API."""
        # Prepare user data
        user_data = {
            "name": "Test User",
            "age": 25,
            "gender": "female",
            "location": "San Francisco",
            "bio": "Love hiking and photography",
            "interests": ["hiking", "photography", "travel"]
        }
        
        # Make API request
        response = await async_client.post("/api/v1/users/", json=user_data)
        
        # Verify response
        assert response.status_code == 201
        response_data = response.json()
        assert response_data["name"] == user_data["name"]
        assert response_data["age"] == user_data["age"]
        assert response_data["gender"] == user_data["gender"]
        assert response_data["location"] == user_data["location"]
        assert response_data["bio"] == user_data["bio"]
        assert response_data["interests"] == user_data["interests"]
        assert "id" in response_data
        assert "created_at" in response_data
        
        # Verify database state
        user_id = response_data["id"]
        user = await db_session.get(User, user_id)
        assert user is not None
        assert user.name == user_data["name"]
        assert user.age == user_data["age"]
    
    @pytest.mark.asyncio
    async def test_get_user(self, async_client: AsyncClient, test_user: User):
        """Test getting a user by ID."""
        # Make API request
        response = await async_client.get(f"/api/v1/users/{test_user.id}")
        
        # Verify response
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["id"] == test_user.id
        assert response_data["name"] == test_user.name
        assert response_data["age"] == test_user.age
        assert response_data["gender"] == test_user.gender
        assert response_data["location"] == test_user.location
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_user(self, async_client: AsyncClient):
        """Test getting a non-existent user."""
        response = await async_client.get("/api/v1/users/9999")
        
        # Verify error response
        assert response.status_code == 404
        response_data = response.json()
        assert "User with ID 9999 not found" in response_data["detail"]
    
    @pytest.mark.asyncio
    async def test_update_user(self, async_client: AsyncClient, test_user: User, db_session: AsyncSession):
        """Test updating a user."""
        # Prepare update data
        update_data = {
            "name": "Updated Name",
            "age": 26,
            "bio": "Updated bio"
        }
        
        # Make API request
        response = await async_client.put(f"/api/v1/users/{test_user.id}", json=update_data)
        
        # Verify response
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["name"] == update_data["name"]
        assert response_data["age"] == update_data["age"]
        assert response_data["bio"] == update_data["bio"]
        # Other fields should remain unchanged
        assert response_data["gender"] == test_user.gender
        assert response_data["location"] == test_user.location
        
        # Verify database update
        await db_session.refresh(test_user)
        assert test_user.name == update_data["name"]
        assert test_user.age == update_data["age"]
        assert test_user.bio == update_data["bio"]
    
    @pytest.mark.asyncio
    async def test_update_nonexistent_user(self, async_client: AsyncClient):
        """Test updating a non-existent user."""
        update_data = {"name": "Updated Name"}
        response = await async_client.put("/api/v1/users/9999", json=update_data)
        
        # Verify error response
        assert response.status_code == 404
        assert "User with ID 9999 not found" in response.json()["detail"]
    
    @pytest.mark.asyncio
    async def test_delete_user(self, async_client: AsyncClient, test_user: User, db_session: AsyncSession):
        """Test deleting a user."""
        # Make API request
        response = await async_client.delete(f"/api/v1/users/{test_user.id}")
        
        # Verify response
        assert response.status_code == 204
        
        # Verify database deletion
        await db_session.refresh(test_user)
        user = await db_session.get(User, test_user.id)
        assert user is None
    
    @pytest.mark.asyncio
    async def test_delete_nonexistent_user(self, async_client: AsyncClient):
        """Test deleting a non-existent user."""
        response = await async_client.delete("/api/v1/users/9999")
        
        # Verify error response
        assert response.status_code == 404
        assert "User with ID 9999 not found" in response.json()["detail"]
    
    @pytest.mark.asyncio
    async def test_list_users(self, async_client: AsyncClient, db_session: AsyncSession):
        """Test listing users with pagination."""
        # Create multiple test users
        users_data = [
            {"name": f"User {i}", "age": 20 + i, "gender": "female", "location": "San Francisco"}
            for i in range(5)
        ]
        
        for user_data in users_data:
            user = User(**user_data)
            db_session.add(user)
        await db_session.commit()
        
        # Make API request
        response = await async_client.get("/api/v1/users/?skip=1&limit=3")
        
        # Verify response
        assert response.status_code == 200
        response_data = response.json()
        assert len(response_data) == 3
        assert response_data[0]["name"] == "User 1"
        assert response_data[-1]["name"] == "User 3"
    
    @pytest.mark.asyncio
    async def test_list_users_with_age_filter(self, async_client: AsyncClient, db_session: AsyncSession):
        """Test listing users with age filter."""
        # Create users with different ages
        users_data = [
            {"name": "Young User", "age": 20, "gender": "female", "location": "SF"},
            {"name": "Middle User", "age": 30, "gender": "male", "location": "SF"},
            {"name": "Old User", "age": 40, "gender": "female", "location": "SF"}
        ]
        
        for user_data in users_data:
            user = User(**user_data)
            db_session.add(user)
        await db_session.commit()
        
        # Make API request with age filter
        response = await async_client.get("/api/v1/users/?min_age=25&max_age=35")
        
        # Verify response
        assert response.status_code == 200
        response_data = response.json()
        assert len(response_data) == 1
        assert response_data[0]["name"] == "Middle User"
        assert 25 <= response_data[0]["age"] <= 35
    
    @pytest.mark.asyncio
    async def test_list_users_with_location_filter(self, async_client: AsyncClient, db_session: AsyncSession):
        """Test listing users with location filter."""
        # Create users in different locations
        users_data = [
            {"name": "SF User", "age": 25, "gender": "female", "location": "San Francisco"},
            {"name": "NY User", "age": 30, "gender": "male", "location": "New York"},
            {"name": "LA User", "age": 28, "gender": "female", "location": "Los Angeles"}
        ]
        
        for user_data in users_data:
            user = User(**user_data)
            db_session.add(user)
        await db_session.commit()
        
        # Make API request with location filter
        response = await async_client.get("/api/v1/users/?location=San Francisco")
        
        # Verify response
        assert response.status_code == 200
        response_data = response.json()
        assert len(response_data) == 1
        assert response_data[0]["name"] == "SF User"
        assert "San Francisco" in response_data[0]["location"]
    
    @pytest.mark.asyncio
    async def test_search_active_users(self, async_client: AsyncClient, db_session: AsyncSession):
        """Test searching for active users."""
        # Create users with different last active times
        from datetime import datetime, timedelta
        
        active_user = User(
            name="Active User",
            age=25,
            gender="female",
            location="SF",
            last_active_at=datetime.utcnow() - timedelta(days=5)
        )
        inactive_user = User(
            name="Inactive User",
            age=30,
            gender="male",
            location="SF",
            last_active_at=datetime.utcnow() - timedelta(days=40)
        )
        
        db_session.add(active_user)
        db_session.add(inactive_user)
        await db_session.commit()
        
        # Make API request
        response = await async_client.get("/api/v1/users/search/active?days_since_active=30")
        
        # Verify response
        assert response.status_code == 200
        response_data = response.json()
        assert len(response_data) == 1
        assert response_data[0]["name"] == "Active User"
    
    @pytest.mark.asyncio
    async def test_search_by_interest(self, async_client: AsyncClient, db_session: AsyncSession):
        """Test searching users by interest."""
        # Create users with different interests
        user1 = User(
            name="Hiker",
            age=25,
            gender="female",
            location="SF",
            interests=["hiking", "photography"]
        )
        user2 = User(
            name="Cook",
            age=30,
            gender="male",
            location="SF",
            interests=["cooking", "photography"]
        )
        user3 = User(
            name="Reader",
            age=28,
            gender="female",
            location="SF",
            interests=["reading", "writing"]
        )
        
        db_session.add(user1)
        db_session.add(user2)
        db_session.add(user3)
        await db_session.commit()
        
        # Make API request
        response = await async_client.get("/api/v1/users/search/by-interest?interest=photography")
        
        # Verify response
        assert response.status_code == 200
        response_data = response.json()
        assert len(response_data) == 2
        user_names = [user["name"] for user in response_data]
        assert "Hiker" in user_names
        assert "Cook" in user_names
        assert "Reader" not in user_names
    
    @pytest.mark.asyncio
    async def test_validation_error_on_create(self, async_client: AsyncClient):
        """Test validation error when creating user with invalid data."""
        # Invalid age (too young)
        invalid_data = {
            "name": "Young User",
            "age": 15,  # Invalid - too young
            "gender": "female",
            "location": "SF"
        }
        
        response = await async_client.post("/api/v1/users/", json=invalid_data)
        
        # Verify validation error
        assert response.status_code == 422
        response_data = response.json()
        assert "validation error" in response_data["error"].lower()
    
    @pytest.mark.asyncio
    async def test_empty_interests_list(self, async_client: AsyncClient):
        """Test creating user with empty interests list."""
        user_data = {
            "name": "No Interests User",
            "age": 25,
            "gender": "female",
            "location": "SF",
            "interests": []  # Empty list
        }
        
        response = await async_client.post("/api/v1/users/", json=user_data)
        
        # Should succeed with empty interests
        assert response.status_code == 201
        response_data = response.json()
        assert response_data["interests"] == []
    
    @pytest.mark.asyncio
    async def test_duplicate_interests(self, async_client: AsyncClient):
        """Test that duplicate interests are deduplicated."""
        user_data = {
            "name": "Duplicate Interests User",
            "age": 25,
            "gender": "female",
            "location": "SF",
            "interests": ["hiking", "photography", "hiking", "travel"]  # Duplicate "hiking"
        }
        
        response = await async_client.post("/api/v1/users/", json=user_data)
        
        # Should succeed and deduplicate interests
        assert response.status_code == 201
        response_data = response.json()
        assert len(response_data["interests"]) == 3
        assert response_data["interests"].count("hiking") == 1
        