"""
Test configuration and fixtures.

This module provides common test fixtures and configuration for all test suites.
Design Rationale:
- Centralized test configuration
- Reusable test fixtures
- Database isolation for tests
- Mock services for unit testing
"""

import asyncio
from typing import AsyncGenerator, Generator
import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient
from httpx import AsyncClient

from app.main import app
from app.core.config import get_settings
from app.core.db import SQLModel
from app.models.sql_models import User, Interaction, UserEmbedding, UserFeatures

# Test database URL
TEST_DATABASE_URL = "postgresql+asyncpg://postgres:postgres@localhost:5432/two-stage-recommender_test"

# Create test engine
engine = create_async_engine(TEST_DATABASE_URL, echo=True)
TestingSessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False
)


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def test_db() -> AsyncGenerator[None, None]:
    """Create test database and tables."""
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    
    yield
    
    # Cleanup
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)


@pytest.fixture
async def db_session(test_db) -> AsyncGenerator[AsyncSession, None]:
    """Create a database session for each test."""
    async with TestingSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """Create a test client for the FastAPI app."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """Create an async test client."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def test_user_data() -> dict:
    """Sample user data for testing."""
    return {
        "name": "Test User",
        "age": 25,
        "gender": "female",
        "location": "San Francisco",
        "bio": "Love hiking and photography",
        "interests": ["hiking", "photography", "travel"]
    }


@pytest.fixture
async def test_user(test_user_data: dict, db_session: AsyncSession) -> User:
    """Create a test user."""
    user = User(**test_user_data)
    db_session.add(user)
    await db_session.flush()
    await db_session.refresh(user)
    return user


@pytest.fixture
def another_test_user() -> dict:
    """Another sample user for testing interactions."""
    return {
        "name": "Another User",
        "age": 28,
        "gender": "male",
        "location": "San Francisco",
        "bio": "Software engineer who loves cooking",
        "interests": ["cooking", "technology", "travel"]
    }


@pytest.fixture
def test_interaction_data() -> dict:
    """Sample interaction data for testing."""
    return {
        "user_id": 1,
        "target_user_id": 2,
        "interaction_type": "like",
        "context": {"source": "recommendation", "confidence": 0.85}
    }


@pytest.fixture
def mock_feature_schema() -> dict:
    """Mock feature schema for testing."""
    return {
        "age": "numeric",
        "gender": "categorical",
        "location": "categorical",
        "account_age_days": "numeric",
        "interests_count": "numeric",
        "total_interactions": "numeric",
        "like_rate": "numeric",
        "recent_activity_30d": "numeric"
    }


@pytest.fixture
def mock_user_features() -> dict:
    """Mock user features for testing."""
    return {
        "age": 25,
        "gender": "female",
        "location": "San Francisco",
        "account_age_days": 30,
        "interests_count": 3,
        "total_interactions": 50,
        "like_rate": 0.7,
        "recent_activity_30d": 15
    }
