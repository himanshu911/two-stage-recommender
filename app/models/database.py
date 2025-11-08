"""
Database models using SQLModel ORM.

This module defines the core data models for the dating recommender system.
Design patterns demonstrated:
- Separate domain models from API models
- Type-safe database schemas with SQLModel
- Relationship definitions with lazy loading
- Timestamp tracking for all entities
"""

from datetime import datetime
from typing import Optional, List, TYPE_CHECKING
from enum import Enum

from sqlalchemy import Column, Integer, String, DateTime, Float, JSON, Text, Index
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from sqlmodel import Field, SQLModel, Relationship as SQLRelationship


class InteractionType(str, Enum):
    """Type of user interaction for enum constraint."""
    LIKE = "like"
    DISLIKE = "dislike"
    SUPER_LIKE = "super_like"
    BLOCK = "block"


class User(SQLModel, table=True):
    """
    User entity representing a person in the dating system.
    
    Design Rationale:
    - Using SQLModel for type-safe ORM with Pydantic validation
    - Separating database model from API response models
    - Including timestamp tracking for data lineage
    - Using enum for gender to ensure data consistency
    """
    __tablename__ = "users"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(max_length=100)
    age: int = Field()
    gender: str = Field(max_length=20)
    location: str = Field(max_length=200)
    bio: Optional[str] = Field(default=None, sa_column=Column(Text))
    interests: Optional[List[str]] = Field(default=None, sa_column=Column(JSON))
    
    # Timestamps for data lineage and debugging
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=Column(DateTime, server_default=func.now())
    )
    updated_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime, onupdate=func.now())
    )
    last_active_at: Optional[datetime] = Field(default=None)
    
    # Relationships
    interactions_made: List["Interaction"] = SQLRelationship(
        back_populates="user",
        sa_relationship_kwargs={
            "primaryjoin": "User.id == Interaction.user_id",
            "lazy": "selectin"  # Eager loading for better performance
        }
    )
    
    interactions_received: List["Interaction"] = SQLRelationship(
        back_populates="target_user",
        sa_relationship_kwargs={
            "primaryjoin": "User.id == Interaction.target_user_id",
            "lazy": "selectin"
        }
    )
    
    # Indexes for query performance
    __table_args__ = (
        Index("idx_user_age", "age"),
        Index("idx_user_gender", "gender"),
        Index("idx_user_location", "location"),
        Index("idx_user_created_at", "created_at"),
    )
    
    def __repr__(self) -> str:
        return f"<User(id={self.id}, name={self.name}, age={self.age})>"


class Interaction(SQLModel, table=True):
    """
    User interaction representing likes, dislikes, etc.
    
    Design Rationale:
    - Self-referential many-to-many relationship through interaction table
    - Using enum for interaction type to ensure data consistency
    - Composite index for efficient querying of user interactions
    - Timestamp tracking for temporal analysis
    """
    __tablename__ = "interactions"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="users.id")
    target_user_id: int = Field(foreign_key="users.id")
    interaction_type: InteractionType = Field(max_length=20)
    
    # Additional metadata for ML features
    context: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=Column(DateTime, server_default=func.now(), index=True)
    )
    
    # Relationships
    user: "User" = SQLRelationship(
        back_populates="interactions_made",
        sa_relationship_kwargs={
            "foreign_keys": "[Interaction.user_id]",
            "lazy": "selectin"
        }
    )
    
    target_user: "User" = SQLRelationship(
        back_populates="interactions_received",
        sa_relationship_kwargs={
            "foreign_keys": "[Interaction.target_user_id]",
            "lazy": "selectin"
        }
    )
    
    # Indexes for query performance
    __table_args__ = (
        # Composite index for efficient user interaction queries
        Index("idx_interaction_user_timestamp", "user_id", "timestamp"),
        Index("idx_interaction_target_user", "target_user_id"),
        Index("idx_interaction_type", "interaction_type"),
        # Ensure unique interactions to prevent duplicates
        Index("idx_unique_interaction", "user_id", "target_user_id", unique=True),
    )
    
    def __repr__(self) -> str:
        return f"<Interaction(user_id={self.user_id}, target_user_id={self.target_user_id}, type={self.interaction_type})>"


class UserEmbedding(SQLModel, table=True):
    """
    User embeddings for collaborative filtering.

    Design Rationale:
    - Separate table for ML features to keep user table clean
    - Version tracking for model updates and rollbacks
    - Using JSON for flexible embedding storage
    - Indexing for efficient similarity search
    """
    __tablename__ = "user_embeddings"

    model_config = {"protected_namespaces": ()}
    
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(
        foreign_key="users.id",
        unique=True
    )
    embedding_vector: Optional[List[float]] = Field(default=None, sa_column=Column(JSON))
    model_version: str = Field(
        max_length=50
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=Column(DateTime, server_default=func.now())
    )
    
    # Indexes for performance
    __table_args__ = (
        Index("idx_embedding_user", "user_id", unique=True),
        Index("idx_embedding_model_version", "model_version"),
    )
    
    def __repr__(self) -> str:
        return f"<UserEmbedding(user_id={self.user_id}, model_version={self.model_version})>"


class UserFeatures(SQLModel, table=True):
    """
    Pre-computed user features for ML models.
    
    Design Rationale:
    - Feature store pattern for ML feature management
    - Versioning for feature schema evolution
    - JSON storage for flexible feature sets
    - Efficient lookup by user_id with latest features
    """
    __tablename__ = "user_features"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(
        foreign_key="users.id"
    )
    feature_set: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    computed_at: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=Column(DateTime, server_default=func.now())
    )
    version: str = Field(
        max_length=50
    )
    
    # Indexes for performance
    __table_args__ = (
        Index("idx_features_user_version", "user_id", "version"),
        Index("idx_features_computed_at", "computed_at"),
    )
    
    def __repr__(self) -> str:
        return f"<UserFeatures(user_id={self.user_id}, version={self.version})>"


class MLModel(SQLModel, table=True):
    """
    ML model metadata and storage.

    Design Rationale:
    - Model versioning for A/B testing and rollbacks
    - Metrics storage for model performance tracking
    - Binary storage for model artifacts
    - Metadata for model lineage and reproducibility
    """
    __tablename__ = "ml_models"

    model_config = {"protected_namespaces": ()}
    
    id: Optional[int] = Field(default=None, primary_key=True)
    model_type: str = Field(
        max_length=50
    )  # e.g., 'candidate_generation', 'ranking'
    version: str = Field(
        max_length=50
    )
    model_binary: bytes = Field(
        sa_column=Column(JSON, nullable=False)
    )  # Base64 encoded model
    metrics: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    hyperparameters: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=Column(DateTime, server_default=func.now())
    )
    is_active: bool = Field(default=False)
    
    # Indexes for performance
    __table_args__ = (
        Index("idx_model_type_version", "model_type", "version", unique=True),
        Index("idx_model_active", "is_active"),
    )
    
    def __repr__(self) -> str:
        return f"<MLModel(type={self.model_type}, version={self.version}, active={self.is_active})>"
    