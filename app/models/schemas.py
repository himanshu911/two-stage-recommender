"""
API request/response schemas using Pydantic.

This module defines the API contract models, separate from database models.
Design patterns demonstrated:
- Separate API models from database models for flexibility
- Request/response validation with Pydantic
- Versioned API schemas
- DTO pattern for data transformation
"""

from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class InteractionType(str, Enum):
    """API-level interaction type enum."""
    LIKE = "like"
    DISLIKE = "dislike"
    SUPER_LIKE = "super_like"
    BLOCK = "block"


class UserCreateRequest(BaseModel):
    """
    Request schema for user creation.
    
    Design Rationale:
    - Separate creation schema from update schema
    - Field validation for data quality
    - Clear API contract with documentation
    """
    name: str = Field(..., min_length=1, max_length=100, description="User's display name")
    age: int = Field(..., ge=18, le=120, description="User's age")
    gender: str = Field(..., pattern="^(male|female|other)$", description="User's gender")
    location: str = Field(..., min_length=1, max_length=200, description="User's location")
    bio: Optional[str] = Field(None, max_length=1000, description="User's biography")
    interests: List[str] = Field(default=[], max_items=50, description="User's interests")
    
    @field_validator('interests')
    @classmethod
    def validate_interests(cls, v: List[str]) -> List[str]:
        """Validate that interests are unique and non-empty."""
        if not all(interest.strip() for interest in v):
            raise ValueError("All interests must be non-empty")
        return list(set(interest.strip() for interest in v))


class UserUpdateRequest(BaseModel):
    """Request schema for user updates - allows partial updates."""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    age: Optional[int] = Field(None, ge=18, le=120)
    location: Optional[str] = Field(None, min_length=1, max_length=200)
    bio: Optional[str] = Field(None, max_length=1000)
    interests: Optional[List[str]] = Field(None, max_items=50)
    
    @field_validator('interests')
    @classmethod
    def validate_interests(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is not None:
            if not all(interest.strip() for interest in v):
                raise ValueError("All interests must be non-empty")
            return list(set(interest.strip() for interest in v))
        return v


class UserResponse(BaseModel):
    """
    Response schema for user data.
    
    Design Rationale:
    - Excludes sensitive fields (passwords, internal IDs)
    - Includes computed fields for API convenience
    - Clear separation between internal and external representation
    """
    id: int = Field(..., description="User's unique identifier")
    name: str = Field(..., description="User's display name")
    age: int = Field(..., description="User's age")
    gender: str = Field(..., description="User's gender")
    location: str = Field(..., description="User's location")
    bio: Optional[str] = Field(None, description="User's biography")
    interests: List[str] = Field(default_factory=list, description="User's interests")
    created_at: datetime = Field(..., description="Account creation timestamp")
    last_active_at: Optional[datetime] = Field(None, description="Last activity timestamp")
    match_score: Optional[float] = Field(None, description="Recommendation score (0-1)")

    @field_validator('interests', mode='before')
    @classmethod
    def validate_interests_none(cls, v):
        """Convert None to empty list for ORM compatibility (F7 fix)."""
        return v if v is not None else []

    class Config:
        from_attributes = True  # Enable ORM mode for SQLModel compatibility


class InteractionCreateRequest(BaseModel):
    """Request schema for creating an interaction."""
    user_id: int = Field(..., description="ID of the user creating the interaction")
    target_user_id: int = Field(..., description="ID of the user being interacted with")
    interaction_type: InteractionType = Field(..., description="Type of interaction")
    context: Optional[Dict[str, Any]] = Field(
        default={},
        description="Additional context for the interaction"
    )
    
    @field_validator('target_user_id')
    @classmethod
    def validate_target_user_id(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("target_user_id must be positive")
        return v

    @field_validator('user_id')
    @classmethod
    def validate_user_id(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("user_id must be positive")
        return v


class InteractionResponse(BaseModel):
    """Response schema for interaction data."""
    id: int = Field(..., description="Interaction unique identifier")
    user_id: int = Field(..., description="ID of the user who made the interaction")
    target_user_id: int = Field(..., description="ID of the target user")
    interaction_type: InteractionType = Field(..., description="Type of interaction")
    timestamp: datetime = Field(..., description="When the interaction occurred")
    context: Dict[str, Any] = Field(default={}, description="Interaction context")
    
    class Config:
        from_attributes = True


class RecommendationRequest(BaseModel):
    """Request schema for getting recommendations."""
    limit: int = Field(default=20, ge=1, le=100, description="Number of recommendations to return")
    exclude_seen: bool = Field(default=True, description="Whether to exclude users already seen")
    filters: Optional[Dict[str, Any]] = Field(
        default={},
        description="Additional filters (age_range, location, etc.)"
    )


class RecommendationResponse(BaseModel):
    """Response schema for recommendation results."""
    recommendations: List[UserResponse] = Field(..., description="List of recommended users")
    total_count: int = Field(..., description="Total number of recommendations available")
    algorithm_version: str = Field(..., description="Version of the recommendation algorithm used")
    generation_time_ms: float = Field(..., description="Time taken to generate recommendations in milliseconds")


class HealthCheckResponse(BaseModel):
    """Response schema for health check endpoint."""
    status: str = Field(..., description="Overall system status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    version: str = Field(..., description="Application version")
    checks: Dict[str, str] = Field(..., description="Individual component health statuses")
    
    
class ModelInfoResponse(BaseModel):
    """Response schema for ML model information."""
    model_config = {"protected_namespaces": (), "from_attributes": True}

    model_type: str = Field(..., description="Type of ML model")
    version: str = Field(..., description="Model version")
    is_active: bool = Field(..., description="Whether the model is currently active")
    metrics: Dict[str, Any] = Field(..., description="Model performance metrics")
    created_at: datetime = Field(..., description="Model creation timestamp")


class ErrorResponse(BaseModel):
    """Standard error response schema."""
    error: str = Field(..., description="Error message")
    detail: Optional[Any] = Field(None, description="Detailed error information")
    request_id: Optional[str] = Field(None, description="Request ID for debugging")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Error timestamp")


class TimelineEntry(BaseModel):
    """Entry in an interaction timeline."""
    date: str = Field(..., description="Date of the interval (ISO format)")
    interaction_type: InteractionType = Field(..., description="Type of interaction")
    count: int = Field(..., description="Count of interactions in this interval")


class ExplanationEntry(BaseModel):
    """Explanation for a single recommendation."""
    user_id: int = Field(..., description="ID of the recommended user")
    score: float = Field(..., description="Recommendation score")
    reason: List[str] = Field(..., description="List of reasons for the recommendation")
    contributing_features: Dict[str, float] = Field(..., description="Features contributing to the score")


class ExplanationResponse(BaseModel):
    """Response schema for recommendations with explanations."""
    recommendations: List[UserResponse] = Field(..., description="List of recommended users")
    explanations: Dict[int, ExplanationEntry] = Field(..., description="Map of user ID to explanation")
    total_count: int = Field(..., description="Total number of recommendations")


class PerformanceMetrics(BaseModel):
    """Performance metrics for the system."""
    latency_p50: float = Field(..., description="50th percentile latency in ms")
    latency_p95: float = Field(..., description="95th percentile latency in ms")
    latency_p99: float = Field(..., description="99th percentile latency in ms")
    throughput_rps: float = Field(..., description="Requests per second")
    cache_hit_rate: float = Field(..., description="Cache hit rate (0-1)")
    error_rate: float = Field(..., description="Error rate (0-1)")
    model_loading_time_ms: float = Field(..., description="Model loading time in ms")
    active_models: Dict[str, str] = Field(..., description="Map of model types to active versions")
    