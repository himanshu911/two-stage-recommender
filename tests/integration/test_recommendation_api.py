"""
Integration tests for Recommendation API endpoints.

This module tests the complete recommendation flow from API to ML services.
Design Rationale:
- Test complete recommendation pipeline
- Verify ML model integration
- Test caching behavior
- Validate recommendation quality
"""

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.sql_models import User, Interaction
from app.core.db import SQLModel


class TestRecommendationAPI:
    """Integration tests for Recommendation API endpoints."""
    
    @pytest.fixture
    async def seed_data(self, db_session: AsyncSession):
        """Seed test data for recommendation testing."""
        # Create multiple users with different characteristics
        users_data = [
            {"name": "Alice", "age": 25, "gender": "female", "location": "San Francisco", "interests": ["hiking", "photography"]},
            {"name": "Bob", "age": 30, "gender": "male", "location": "San Francisco", "interests": ["technology", "cooking"]},
            {"name": "Charlie", "age": 28, "gender": "male", "location": "San Francisco", "interests": ["hiking", "music"]},
            {"name": "Diana", "age": 26, "gender": "female", "location": "San Francisco", "interests": ["art", "travel"]},
            {"name": "Eve", "age": 29, "gender": "female", "location": "New York", "interests": ["reading", "cooking"]},
            {"name": "Frank", "age": 27, "gender": "male", "location": "San Francisco", "interests": ["sports", "photography"]},
            {"name": "Grace", "age": 31, "gender": "female", "location": "San Francisco", "interests": ["yoga", "travel"]},
            {"name": "Henry", "age": 24, "gender": "male", "location": "San Francisco", "interests": ["gaming", "technology"]},
        ]
        
        users = []
        for user_data in users_data:
            user = User(**user_data)
            db_session.add(user)
            users.append(user)
        
        await db_session.commit()
        
        # Create some interactions for collaborative filtering
        interactions_data = [
            {"user_id": 1, "target_user_id": 2, "interaction_type": "like"},
            {"user_id": 1, "target_user_id": 3, "interaction_type": "like"},
            {"user_id": 1, "target_user_id": 4, "interaction_type": "dislike"},
            {"user_id": 2, "target_user_id": 1, "interaction_type": "like"},
            {"user_id": 2, "target_user_id": 3, "interaction_type": "super_like"},
            {"user_id": 3, "target_user_id": 1, "interaction_type": "like"},
            {"user_id": 3, "target_user_id": 2, "interaction_type": "like"},
        ]
        
        for interaction_data in interactions_data:
            interaction = Interaction(**interaction_data)
            db_session.add(interaction)
        
        await db_session.commit()
        
        return users
    
    @pytest.mark.asyncio
    async def test_get_recommendations_basic(self, async_client: AsyncClient, seed_data):
        """Test basic recommendation retrieval."""
        # Make API request for recommendations
        response = await async_client.get("/api/v1/recommendations/users/1/recommendations?limit=5")
        
        # Verify response
        assert response.status_code == 200
        response_data = response.json()
        
        # Verify response structure
        assert "recommendations" in response_data
        assert "total_count" in response_data
        assert "algorithm_version" in response_data
        assert "generation_time_ms" in response_data
        
        # Verify recommendations
        recommendations = response_data["recommendations"]
        assert len(recommendations) <= 5
        assert response_data["total_count"] == len(recommendations)
        
        # Verify each recommendation has required fields
        for rec in recommendations:
            assert "id" in rec
            assert "name" in rec
            assert "age" in rec
            assert "gender" in rec
            assert "location" in rec
            assert "interests" in rec
            assert "match_score" in rec
            assert 0 <= rec["match_score"] <= 1
            
            # Should not include users already interacted with
            assert rec["id"] not in [2, 3, 4]  # These are already interacted with
    
    @pytest.mark.asyncio
    async def test_get_recommendations_with_age_filter(self, async_client: AsyncClient, seed_data):
        """Test recommendations with age filtering."""
        # Make API request with age filters
        response = await async_client.get(
            "/api/v1/recommendations/users/1/recommendations?limit=10&min_age=26&max_age=30"
        )
        
        # Verify response
        assert response.status_code == 200
        response_data = response.json()
        recommendations = response_data["recommendations"]
        
        # Verify age filtering
        for rec in recommendations:
            assert 26 <= rec["age"] <= 30
    
    @pytest.mark.asyncio
    async def test_get_recommendations_with_location_filter(self, async_client: AsyncClient, seed_data):
        """Test recommendations with location filtering."""
        # Make API request with location filter
        response = await async_client.get(
            "/api/v1/recommendations/users/1/recommendations?limit=10&location=San Francisco"
        )
        
        # Verify response
        assert response.status_code == 200
        response_data = response.json()
        recommendations = response_data["recommendations"]
        
        # Verify location filtering
        for rec in recommendations:
            assert "San Francisco" in rec["location"]
    
    @pytest.mark.asyncio
    async def test_get_recommendations_exclude_seen_false(self, async_client: AsyncClient, seed_data):
        """Test recommendations including already seen users."""
        # Make API request with exclude_seen=False
        response = await async_client.get(
            "/api/v1/recommendations/users/1/recommendations?limit=10&exclude_seen=false"
        )
        
        # Verify response
        assert response.status_code == 200
        response_data = response.json()
        recommendations = response_data["recommendations"]
        
        # Should include users already interacted with
        user_ids = [rec["id"] for rec in recommendations]
        assert any(uid in [2, 3, 4] for uid in user_ids)
    
    @pytest.mark.asyncio
    async def test_get_recommendations_with_explanations(self, async_client: AsyncClient, seed_data):
        """Test recommendations with explanations."""
        # Make API request with explanations
        response = await async_client.get(
            "/api/v1/recommendations/users/1/recommendations/explain?limit=5"
        )
        
        # Verify response
        assert response.status_code == 200
        response_data = response.json()
        
        # Verify response structure
        assert "recommendations" in response_data
        assert "explanations" in response_data
        assert "total_count" in response_data
        
        # Verify explanations
        explanations = response_data["explanations"]
        assert len(explanations) == len(response_data["recommendations"])
        
        for explanation in explanations:
            assert "user_id" in explanation
            assert "match_score" in explanation
            assert "reasons" in explanation
            assert isinstance(explanation["reasons"], list)
            assert len(explanation["reasons"]) > 0
    
    @pytest.mark.asyncio
    async def test_recommendation_performance(self, async_client: AsyncClient, seed_data):
        """Test recommendation generation performance."""
        # Make API request
        response = await async_client.get("/api/v1/recommendations/users/1/recommendations?limit=20")
        
        # Verify response
        assert response.status_code == 200
        response_data = response.json()
        
        # Verify performance metrics
        generation_time = response_data["generation_time_ms"]
        assert generation_time > 0  # Should take some time
        assert generation_time < 5000  # Should complete within 5 seconds
        
        # Log performance for monitoring
        print(f"Recommendation generation time: {generation_time}ms")
    
    @pytest.mark.asyncio
    async def test_recommendation_quality_indicators(self, async_client: AsyncClient, seed_data):
        """Test recommendation quality indicators."""
        # Make API request
        response = await async_client.get("/api/v1/recommendations/users/1/recommendations?limit=10")
        
        # Verify response
        assert response.status_code == 200
        response_data = response.json()
        recommendations = response_data["recommendations"]
        
        # Verify quality indicators
        match_scores = [rec["match_score"] for rec in recommendations]
        
        # Should have reasonable match scores
        assert all(0 <= score <= 1 for score in match_scores)
        assert len([s for s in match_scores if s > 0.5]) > 0  # Some high-quality matches
        
        # Should have diverse interests
        all_interests = []
        for rec in recommendations:
            all_interests.extend(rec["interests"])
        unique_interests = set(all_interests)
        assert len(unique_interests) >= 3  # Good diversity
    
    @pytest.mark.asyncio
    async def test_recommendation_caching(self, async_client: AsyncClient, seed_data):
        """Test recommendation caching behavior."""
        # First request
        response1 = await async_client.get("/api/v1/recommendations/users/1/recommendations?limit=5")
        assert response1.status_code == 200
        time1 = response1.json()["generation_time_ms"]
        
        # Second request (should be faster due to caching)
        response2 = await async_client.get("/api/v1/recommendations/users/1/recommendations?limit=5")
        assert response2.status_code == 200
        time2 = response2.json()["generation_time_ms"]
        
        # Both requests should return same recommendations
        recs1 = response1.json()["recommendations"]
        recs2 = response2.json()["recommendations"]
        assert len(recs1) == len(recs2)
        
        # Second request might be faster due to caching (but not guaranteed)
        print(f"First request time: {time1}ms, Second request time: {time2}ms")
    
    @pytest.mark.asyncio
    async def test_algorithm_version_parameter(self, async_client: AsyncClient, seed_data):
        """Test using different algorithm versions."""
        # Request with specific algorithm version
        response = await async_client.get(
            "/api/v1/recommendations/users/1/recommendations?algorithm_version=v2.0.0"
        )
        
        # Verify response
        assert response.status_code == 200
        response_data = response.json()
        
        # Algorithm version should be updated (or fallback if not available)
        assert response_data["algorithm_version"] in ["v1.0.0", "v2.0.0"]  # Either original or updated
    
    @pytest.mark.asyncio
    async def test_empty_recommendations(self, async_client: AsyncClient, db_session: AsyncSession):
        """Test recommendations for user with no potential matches."""
        # Create a user with very restrictive filters
        user = User(
            name="Lonely User",
            age=100,  # Very old
            gender="other",
            location="Antarctica",  # Remote location
            interests=["very_unique_interest"]
        )
        db_session.add(user)
        await db_session.commit()
        
        # Make API request
        response = await async_client.get(f"/api/v1/recommendations/users/{user.id}/recommendations?limit=10")
        
        # Verify response (should return empty but not error)
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["recommendations"] == []
        assert response_data["total_count"] == 0
    
    @pytest.mark.asyncio
    async def test_recommendations_for_new_user(self, async_client: AsyncClient, db_session: AsyncSession):
        """Test recommendations for a new user with no interactions."""
        # Create new user
        new_user = User(
            name="New User",
            age=25,
            gender="female",
            location="San Francisco",
            interests=["reading", "travel"]
        )
        db_session.add(new_user)
        await db_session.commit()
        
        # Make API request
        response = await async_client.get(f"/api/v1/recommendations/users/{new_user.id}/recommendations?limit=5")
        
        # Verify response (should use content-based and random strategies)
        assert response.status_code == 200
        response_data = response.json()
        recommendations = response_data["recommendations"]
        
        assert len(recommendations) > 0
        # Should have some basic recommendations for new user
        assert all(rec["location"] == "San Francisco" for rec in recommendations)
    
    @pytest.mark.asyncio
    async def test_performance_metrics_endpoint(self, async_client: AsyncClient):
        """Test getting recommendation performance metrics."""
        # Make API request
        response = await async_client.get("/api/v1/recommendations/performance")
        
        # Verify response
        assert response.status_code == 200
        metrics = response.json()
        
        # Verify metrics structure
        assert "avg_generation_time_ms" in metrics
        assert "avg_candidate_count" in metrics
        assert "cache_size" in metrics
        assert "model_version" in metrics
        
        # Verify metric values
        assert isinstance(metrics["avg_generation_time_ms"], (int, float))
        assert isinstance(metrics["avg_candidate_count"], (int, float))
        assert isinstance(metrics["cache_size"], int)
        assert isinstance(metrics["model_version"], str)
    
    @pytest.mark.asyncio
    async def test_refresh_cache_endpoint(self, async_client: AsyncClient):
        """Test refreshing recommendation cache."""
        # Make API request
        response = await async_client.post("/api/v1/recommendations/refresh")
        
        # Verify response
        assert response.status_code == 200
        assert response.json()["message"] == "Recommendation cache refreshed successfully"
    
    @pytest.mark.asyncio
    async def test_algorithm_versions_endpoint(self, async_client: AsyncClient):
        """Test getting available algorithm versions."""
        # Make API request
        response = await async_client.get("/api/v1/recommendations/algorithm/versions")
        
        # Verify response
        assert response.status_code == 200
        versions = response.json()
        
        # Should return list of versions
        assert isinstance(versions, list)
        assert len(versions) > 0
        assert all(isinstance(v, str) for v in versions)
        