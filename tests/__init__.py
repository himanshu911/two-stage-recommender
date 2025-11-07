"""
Test module initialization.

This module exports test utilities and common test configurations.
"""

# Test utilities and helper functions can be added here
# For example: test data generators, assertion helpers, etc.

from typing import Dict, Any, List


def generate_test_users(count: int) -> List[Dict[str, Any]]:
    """Generate test user data."""
    return [
        {
            "name": f"Test User {i}",
            "age": 20 + i,
            "gender": "female" if i % 2 == 0 else "male",
            "location": "San Francisco",
            "bio": f"Test bio for user {i}",
            "interests": [f"interest_{i}", f"interest_{i+1}"]
        }
        for i in range(count)
    ]


def generate_test_interactions(user_id: int, target_ids: List[int]) -> List[Dict[str, Any]]:
    """Generate test interaction data."""
    import random
    interaction_types = ["like", "dislike", "super_like"]
    
    return [
        {
            "user_id": user_id,
            "target_user_id": target_id,
            "interaction_type": random.choice(interaction_types)
        }
        for target_id in target_ids
    ]


def assert_valid_user_response(user_data: Dict[str, Any]) -> None:
    """Validate user response data."""
    required_fields = ["id", "name", "age", "gender", "location", "created_at"]
    for field in required_fields:
        assert field in user_data, f"Missing required field: {field}"
    
    assert isinstance(user_data["id"], int)
    assert isinstance(user_data["name"], str)
    assert isinstance(user_data["age"], int)
    assert user_data["age"] >= 18
    assert user_data["gender"] in ["male", "female", "other"]


def assert_valid_recommendation_response(response_data: Dict[str, Any]) -> None:
    """Validate recommendation response data."""
    required_fields = ["recommendations", "total_count", "algorithm_version", "generation_time_ms"]
    for field in required_fields:
        assert field in response_data, f"Missing required field: {field}"
    
    assert isinstance(response_data["recommendations"], list)
    assert isinstance(response_data["total_count"], int)
    assert isinstance(response_data["generation_time_ms"], (int, float))
    assert response_data["generation_time_ms"] > 0
    
    for rec in response_data["recommendations"]:
        assert_valid_user_response(rec)
        assert "match_score" in rec
        assert 0 <= rec["match_score"] <= 1
        