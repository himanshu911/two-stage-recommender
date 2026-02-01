"""
Application configuration management using Pydantic Settings.

This module demonstrates production patterns for configuration management:
- Environment-based configuration separation
- Type-safe configuration with validation
- Default values with environment variable overrides
- Secrets management through environment variables
"""

from functools import lru_cache
from typing import Optional, List, Any

from pydantic import model_validator, PostgresDsn
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings with environment variable support.
    
    Design Rationale:
    - Using Pydantic Settings for automatic environment variable parsing
    - Type hints ensure configuration is validated at startup
    - LRU cache prevents repeated file I/O for settings access
    - Environment-based configuration supports 12-factor app principles
    """
    
    # Application
    APP_NAME: str = "Dating Recommender System"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4
    
    # Database
    DATABASE_URL: Optional[PostgresDsn] = None
    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 20
    DATABASE_POOL_PRE_PING: bool = True
    
    # Redis for caching
    REDIS_URL: str = "redis://localhost:6379"
    
    # ML Model Storage
    MODEL_STORAGE_PATH: str = "/app/models"
    FEATURE_STORE_URL: str = "postgresql://user:password@localhost:5432/features"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    
    # ML Configuration
    EMBEDDING_DIMENSION: int = 64
    CANDIDATE_GENERATION_TOP_K: int = 100
    RANKING_MODEL_VERSION: str = "v1.0.0"
    
    # API Configuration
    API_V1_STR: str = "/api/v1"
    CORS_ORIGINS: List[str] = ["*"]
    
    @model_validator(mode='before')
    @classmethod
    def build_database_url(cls, data: dict) -> dict:
        """
        Build database URL from components if not provided directly.
        This pattern allows both direct URL and component-based configuration.
        """
        if data.get("DATABASE_URL") is None:
            # Fallback to building from components
            user = data.get("POSTGRES_USER", "postgres")
            password = data.get("POSTGRES_PASSWORD", "postgres")
            host = data.get("POSTGRES_HOST", "localhost")
            port = data.get("POSTGRES_PORT", 5432)
            db = data.get("POSTGRES_DB", "two-stage-recommender")

            data["DATABASE_URL"] = f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{db}"

        return data
    
    # Configuration for Pydantic v2 Settings (using nested Config class for compatibility)
    class Config:
        # Load local dev defaults from .env (if present).
        # Real environment variables still take precedence (production-friendly).
        # class defaults are last
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Design Rationale:
    - Using LRU cache to prevent repeated file I/O
    - Singleton pattern ensures consistent configuration across app
    - Cache makes configuration access effectively O(1)
    - Note: singleton per process, N workers means N process and N settings instances
    """
    return Settings()


# Global settings instance for convenience
settings = get_settings()
