"""
API request/response schemas using Pydantic.

This module defines the API contract models, separate from database models.
Design patterns demonstrated:
- Separate API models from database models for flexibility
- Request/response validation with Pydantic
- Versioned API schemas
- DTO pattern for data transformation
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum

from pydantic import BaseModel, Field, validator


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
    
    @validator('interests')
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
    
    @validator('interests')
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
    interests: List[str] = Field(default=[], description="User's interests")
    created_at: datetime = Field(..., description="Account creation timestamp")
    last_active_at: Optional[datetime] = Field(None, description="Last activity timestamp")
    match_score: Optional[float] = Field(None, description="Recommendation score (0-1)")
    
    class Config:
        from_attributes = True  # Enable ORM mode for SQLModel compatibility


class InteractionCreateRequest(BaseModel):
    """Request schema for creating an interaction."""
    target_user_id: int = Field(..., description="ID of the user being interacted with")
    interaction_type: InteractionType = Field(..., description="Type of interaction")
    context: Optional[Dict[str, Any]] = Field(
        default={},
        description="Additional context for the interaction"
    )
    
    @validator('target_user_id')
    def validate_target_user_id(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("target_user_id must be positive")
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
    detail: Optional[str] = Field(None, description="Detailed error information")
    request_id: Optional[str] = Field(None, description="Request ID for debugging")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    