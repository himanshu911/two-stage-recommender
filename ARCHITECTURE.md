# Two-Stage Recommender System Architecture

## Overview

This document provides a comprehensive overview of the production-ready two-stage recommender system architecture. The system implements a two-stage recommendation pipeline with collaborative filtering and machine learning-based ranking, designed for scalability, maintainability, and production deployment.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Design Patterns](#design-patterns)
3. [Technology Stack](#technology-stack)
4. [Data Models](#data-models)
5. [Service Architecture](#service-architecture)
6. [ML Pipeline](#ml-pipeline)
7. [API Design](#api-design)
8. [Production Considerations](#production-considerations)
9. [Deployment](#deployment)
10. [Monitoring and Observability](#monitoring-and-observability)

## System Architecture

### High-Level Architecture

The system follows a microservices-inspired architecture with the following key components:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   API Gateway   │    │   Load Balancer │    │     Client      │
│   (FastAPI)     │◄──►│                 │◄──►│   (Web/Mobile)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Application Layer                         │
├─────────────────┬─────────────────┬─────────────────────────────┤
│   User Service  │ Interaction Svc │   Recommendation Service     │
│                 │                 │                             │
├─────────────────┼─────────────────┼─────────────────────────────┤
│ Candidate Gen.  │ Ranking Service │   Feature Service           │
│   Service       │                 │                             │
└─────────────────┴─────────────────┴─────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Data Access Layer                           │
├─────────────────┬─────────────────┬─────────────────────────────┤
│ User Repository │ Interaction Repo│   Feature Repository        │
│                 │                 │                             │
├─────────────────┼─────────────────┼─────────────────────────────┤
│   PostgreSQL    │     Redis       │   Feature Store (PostgreSQL)│
│   (Primary DB)  │    (Cache)      │                             │
└─────────────────┴─────────────────┴─────────────────────────────┘
```

### Core Design Principles

1. **Separation of Concerns**: Each component has a single responsibility
2. **Dependency Injection**: Loose coupling between components for testability
3. **Repository Pattern**: Abstract data access for maintainability
4. **Service Layer**: Business logic separated from API layer
5. **Async/Await**: High-performance I/O operations
6. **Feature Store**: Centralized feature management for ML consistency

## Design Patterns

### Repository Pattern

**Purpose**: Abstract data access logic from business logic

**Implementation**:
```python
class UserRepository(SQLModelRepository[User]):
    """Repository for User entity with domain-specific methods."""
    
    async def get_by_location(self, location: str, skip: int = 0, limit: int = 100) -> List[User]:
        """Get users by location with pagination."""
        # Implementation details...
```

**Benefits**:
- Testability: Easy to mock repositories in unit tests
- Maintainability: Data access logic centralized
- Flexibility: Can switch data sources without affecting business logic

### Service Layer Pattern

**Purpose**: Encapsulate business logic and coordinate between repositories

**Implementation**:
```python
class RecommendationService:
    """Main service for generating recommendations."""
    
    async def get_recommendations(self, user_id: int, limit: int = 20) -> RecommendationResponse:
        """Orchestrate the two-stage recommendation pipeline."""
        # Step 1: Generate candidates
        candidates = await self.candidate_generation_service.generate_candidates(user_id, limit)
        
        # Step 2: Rank candidates
        ranked_candidates = await self.ranking_service.rank_candidates(user_id, candidates)
        
        # Step 3: Return formatted response
        return self._build_response(ranked_candidates)
```

**Benefits**:
- Transaction boundaries: Services manage database transactions
- Business logic centralization: All business rules in one place
- API abstraction: Clean interface for API layer

### Dependency Injection

**Purpose**: Manage dependencies and enable loose coupling

**Implementation**:
```python
async def get_recommendations(
    user_id: int,
    recommendation_service: RecommendationService = Depends()
) -> RecommendationResponse:
    """Get personalized recommendations for a user."""
    return await recommendation_service.get_recommendations(user_id)
```

**Benefits**:
- Testability: Easy to inject mock services for testing
- Flexibility: Can change service implementations without changing API
- Lifecycle management: Automatic resource cleanup

### Two-Stage Recommendation

**Purpose**: Efficient and accurate recommendation generation

**Stage 1 - Candidate Generation**:
- Collaborative filtering using matrix factorization
- Content-based filtering using user features
- Random exploration for diversity
- Efficient similarity search using FAISS

**Stage 2 - Ranking**:
- Logistic regression model for probability scoring
- Real-time feature computation
- Multiple ranking strategies (A/B testing ready)
- Feature importance tracking

**Benefits**:
- Scalability: Can handle large user bases efficiently
- Accuracy: Combines multiple recommendation strategies
- Real-time: Fast inference for online recommendations
- Flexibility: Easy to add new ranking models

## Technology Stack

### Web Framework
- **FastAPI**: Modern, fast web framework for building APIs
  - Async/await support for high performance
  - Automatic API documentation
  - Type hints for validation
  - Dependency injection system

### Database
- **PostgreSQL**: Primary database for user and interaction data
  - ACID compliance for data consistency
  - JSON support for flexible schemas
  - Full-text search capabilities
  - Materialized views for performance

- **Redis**: Caching layer for recommendations and features
  - In-memory performance
  - TTL support for cache expiration
  - Pub/sub for real-time updates

### Machine Learning
- **Scikit-learn**: ML algorithms and utilities
  - Logistic regression for ranking
  - Feature preprocessing
  - Model evaluation metrics

- **FAISS**: Efficient similarity search
  - Vector similarity search
  - Optimized for large datasets
  - GPU acceleration support

- **PyTorch**: Deep learning framework
  - Matrix factorization models
  - Custom embedding architectures
  - GPU acceleration

### Infrastructure
- **Docker**: Containerization for deployment
- **Docker Compose**: Local development environment
- **SQLModel**: Type-safe ORM with Pydantic integration
- **Alembic**: Database migration management

## Data Models

### User Entity
```python
class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(sa_column=Column(String(100), nullable=False))
    age: int = Field(sa_column=Column(Integer, nullable=False))
    gender: str = Field(sa_column=Column(String(20), nullable=False))
    location: str = Field(sa_column=Column(String(200), nullable=False))
    bio: Optional[str] = Field(default=None, sa_column=Column(Text))
    interests: List[str] = Field(default=[], sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = Field(default=None)
    last_active_at: Optional[datetime] = Field(default=None)
```

### Interaction Entity
```python
class Interaction(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="users.id")
    target_user_id: int = Field(foreign_key="users.id")
    interaction_type: InteractionType = Field(sa_column=Column(String(20), nullable=False))
    context: dict = Field(default={}, sa_column=Column(JSON))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
```

### Feature Store Entities
```python
class UserEmbedding(SQLModel, table=True):
    user_id: int = Field(foreign_key="users.id", unique=True)
    embedding_vector: List[float] = Field(sa_column=Column(JSON, nullable=False))
    model_version: str = Field(sa_column=Column(String(50), nullable=False))
    created_at: datetime = Field(default_factory=datetime.utcnow)

class UserFeatures(SQLModel, table=True):
    user_id: int = Field(foreign_key="users.id")
    feature_set: dict = Field(sa_column=Column(JSON, nullable=False))
    computed_at: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field(sa_column=Column(String(50), nullable=False))
```

## Service Architecture

### Feature Service

**Purpose**: Centralized feature computation and management

**Key Features**:
- Real-time feature extraction
- Feature versioning
- Training-serving consistency
- Performance optimization with caching

**Feature Types**:
- **Demographic**: Age, gender, location, account age
- **Behavioral**: Interaction patterns, activity levels, preferences
- **Collaborative**: User embeddings, similarity scores

**Implementation**:
```python
class FeatureService:
    async def get_features(self, user_id: int, feature_types: List[str]) -> Dict[str, Any]:
        """Get features for a user with caching and versioning."""
        # Check cache
        cached_features = await self.get_stored_features(user_id, version)
        if cached_features:
            return cached_features
        
        # Extract features
        features = {}
        for feature_type in feature_types:
            extractor = self.extractors[feature_type]
            features.update(await extractor.extract(user_id))
        
        # Cache features
        await self.store_features(user_id, features, version)
        return features
```

### Candidate Generation Service

**Purpose**: Generate diverse candidate pool for ranking

**Strategies**:
1. **Collaborative Filtering**: Matrix factorization with FAISS similarity search
2. **Content-Based**: User feature similarity matching
3. **Random Exploration**: Diversity and serendipity
4. **Social Graph**: Mutual friends and connections

**Implementation**:
```python
class CandidateGenerationService:
    async def generate_candidates(self, user_id: int, limit: int) -> List[Candidate]:
        """Generate candidates using multiple strategies."""
        candidates = []
        
        # Collaborative filtering
        cf_candidates = await self._generate_cf_candidates(user_id, limit // 2)
        candidates.extend(cf_candidates)
        
        # Content-based
        content_candidates = await self._generate_content_candidates(user_id, limit // 3)
        candidates.extend(content_candidates)
        
        # Random exploration
        random_candidates = await self._generate_random_candidates(user_id, limit - len(candidates))
        candidates.extend(random_candidates)
        
        return self._deduplicate_and_rank(candidates)
```

### Ranking Service

**Purpose**: Score and rank candidates using ML models

**Model Architecture**:
- **Base Model**: Logistic regression for interpretability
- **Features**: User features, interaction history, contextual data
- **Training**: Offline training with online inference
- **A/B Testing**: Multiple model versions support

**Implementation**:
```python
class RankingService:
    async def rank_candidates(self, user_id: int, candidate_ids: List[int]) -> List[RankedCandidate]:
        """Rank candidates using ML model."""
        # Get features for all candidates
        features = await self._get_candidate_features(user_id, candidate_ids)
        
        # Predict probabilities
        probabilities = self.model.predict_proba(features)
        
        # Create ranked candidates
        return [
            RankedCandidate(
                user_id=candidate_id,
                score=probability,
                features=feature_dict,
                model_version=self.model_version
            )
            for candidate_id, probability, feature_dict in zip(candidate_ids, probabilities, features)
        ]
```

## ML Pipeline

### Model Training

**Offline Training**:
1. Data collection and preprocessing
2. Feature engineering and selection
3. Model training and validation
4. Model evaluation and comparison
5. Model deployment and versioning

**Online Training** (Future):
1. Incremental model updates
2. Real-time feedback incorporation
3. Continuous learning
4. Model drift detection

### Feature Engineering

**User Features**:
- Demographics: Age, gender, location, education
- Behavior: Interaction patterns, session data, preferences
- Content: Profile text, interests, photos
- Social: Network connections, mutual friends

**Context Features**:
- Time: Day of week, time of day, seasonality
- Location: Geographic proximity, urban/rural
- Device: Mobile/desktop, app version
- External: Weather, events, trends

### Model Evaluation

**Metrics**:
- **Precision@K**: Relevance of top-K recommendations
- **Recall@K**: Coverage of relevant items
- **NDCG**: Ranking quality
- **CTR**: Click-through rate
- **Conversion Rate**: Successful interactions
- **Diversity**: Recommendation variety
- **Serendipity**: Unexpected but relevant recommendations

## API Design

### RESTful Endpoints

**User Management**:
- `POST /api/v1/users/` - Create user
- `GET /api/v1/users/{user_id}` - Get user
- `PUT /api/v1/users/{user_id}` - Update user
- `DELETE /api/v1/users/{user_id}` - Delete user
- `GET /api/v1/users/` - List users with filtering

**Interactions**:
- `POST /api/v1/interactions/` - Create interaction
- `GET /api/v1/interactions/user/{user_id}` - Get user interactions
- `GET /api/v1/interactions/stats/{user_id}` - Get interaction statistics

**Recommendations**:
- `GET /api/v1/recommendations/users/{user_id}/recommendations` - Get recommendations
- `GET /api/v1/recommendations/users/{user_id}/recommendations/explain` - Get with explanations

**Health and Monitoring**:
- `GET /api/v1/health/` - Comprehensive health check
- `GET /api/v1/health/ready` - Readiness check
- `GET /api/v1/health/live` - Liveness check
- `GET /api/v1/health/metrics` - Application metrics

### API Design Principles

1. **RESTful Design**: Proper HTTP methods and status codes
2. **Pagination**: Consistent pagination with skip/limit parameters
3. **Filtering**: Query parameters for data filtering
4. **Versioning**: URL-based API versioning
5. **Documentation**: Automatic OpenAPI documentation
6. **Error Handling**: Consistent error response format

### Request/Response Examples

**Get Recommendations**:
```http
GET /api/v1/recommendations/users/123/recommendations?limit=20&min_age=25&max_age=35
```

**Response**:
```json
{
  "recommendations": [
    {
      "id": 456,
      "name": "Jane Doe",
      "age": 28,
      "gender": "female",
      "location": "San Francisco",
      "bio": "Love hiking and photography",
      "interests": ["hiking", "photography", "travel"],
      "match_score": 0.85,
      "created_at": "2024-01-15T10:30:00Z",
      "last_active_at": "2024-01-20T15:45:00Z"
    }
  ],
  "total_count": 20,
  "algorithm_version": "v1.0.0",
  "generation_time_ms": 145.67
}
```

## Production Considerations

### Performance Optimization

**Caching Strategy**:
- Redis for recommendation caching (10-minute TTL)
- Feature caching with 5-minute TTL
- Database query result caching
- CDN for static assets

**Database Optimization**:
- Connection pooling (10-20 connections)
- Indexed columns for frequent queries
- Materialized views for complex aggregations
- Read replicas for scaling reads

**ML Optimization**:
- Model quantization for faster inference
- Batch prediction for multiple users
- GPU acceleration for training
- Model versioning for A/B testing

### Security

**Authentication & Authorization**:
- JWT tokens for API authentication
- Role-based access control (RBAC)
- API rate limiting per user
- Request validation and sanitization

**Data Protection**:
- Encryption at rest for sensitive data
- HTTPS/TLS for data in transit
- PII data masking and anonymization
- GDPR compliance features

**Infrastructure Security**:
- Network segmentation
- Firewall rules
- Regular security audits
- Dependency vulnerability scanning

### Scalability

**Horizontal Scaling**:
- Load balancer for API servers
- Database read replicas
- Redis cluster for caching
- Microservices architecture

**Vertical Scaling**:
- CPU optimization for ML inference
- Memory optimization for caching
- I/O optimization for database queries
- Network optimization for external APIs

### Reliability

**Error Handling**:
- Comprehensive exception handling
- Circuit breaker pattern for external services
- Graceful degradation for non-critical features
- Retry mechanisms with exponential backoff

**Monitoring**:
- Health checks for all components
- Performance metrics collection
- Error tracking and alerting
- Log aggregation and analysis

## Deployment

### Docker Configuration

**Multi-stage Build**:
```dockerfile
# Builder stage
FROM python:3.11-slim as builder
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.11-slim
COPY --from=builder /root/.local /home/app/.local
COPY --chown=app:app . /app
USER app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Docker Compose**:
```yaml
version: '3.8'
services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: two-stage-recommender
    volumes:
      - postgres_data:/var/lib/postgresql/data
  
  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
  
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:postgres@postgres:5432/two-stage-recommender
      - REDIS_URL=redis://redis:6379
    depends_on:
      - postgres
      - redis
```

### Environment Configuration

```bash
# Application settings
APP_NAME="Two-Stage Recommender System"
DEBUG=False
ENVIRONMENT=production
HOST=0.0.0.0
PORT=8000
WORKERS=4

# Database
DATABASE_URL=postgresql://user:password@host:5432/database
DATABASE_POOL_SIZE=10
DATABASE_MAX_OVERFLOW=20

# ML Configuration
EMBEDDING_DIMENSION=64
CANDIDATE_GENERATION_TOP_K=100
RANKING_MODEL_VERSION=v1.0.0

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
```

### Deployment Strategies

**Blue-Green Deployment**:
- Zero-downtime deployments
- Easy rollback capability
- A/B testing support
- Gradual traffic shifting

**Canary Deployment**:
- Gradual rollout to subset of users
- Risk mitigation for new features
- Performance monitoring during rollout
- Automatic rollback on issues

## Monitoring and Observability

### Health Checks

**Component Health**:
- Database connectivity
- Redis availability
- Model service status
- External API dependencies

**Health Check Endpoint**:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-20T10:30:00Z",
  "version": "Two-Stage Recommender System",
  "checks": {
    "database": "healthy",
    "redis": "healthy",
    "feature_service": "healthy",
    "recommendation_service": "healthy"
  }
}
```

### Metrics

**Application Metrics**:
- Request rate and latency
- Error rates by endpoint
- Database query performance
- Cache hit rates

**ML Metrics**:
- Model inference time
- Recommendation generation time
- Feature computation time
- Candidate pool size

**Business Metrics**:
- User engagement rates
- Match success rates
- Recommendation acceptance rates
- User retention metrics

### Logging

**Structured Logging**:
```json
{
  "timestamp": "2024-01-20T10:30:00Z",
  "level": "INFO",
  "message": "Recommendations generated",
  "user_id": 123,
  "count": 20,
  "generation_time_ms": 145.67,
  "request_id": "uuid-12345"
}
```

**Log Levels**:
- DEBUG: Detailed debugging information
- INFO: General application flow
- WARNING: Potential issues
- ERROR: Error conditions
- CRITICAL: Critical errors

### Alerting

**Alert Conditions**:
- High error rates (>5%)
- High latency (>500ms p95)
- Database connection failures
- Model inference failures
- Low recommendation quality

**Notification Channels**:
- Email alerts
- Slack notifications
- PagerDuty integration
- Dashboard alerts

### Tracing

**Distributed Tracing**:
- Request flow across services
- Performance bottlenecks identification
- Error propagation tracking
- Dependency mapping

**Implementation**:
- OpenTelemetry integration
- Jaeger for trace collection
- Grafana for visualization
- Custom spans for ML operations

## Conclusion

This architecture provides a robust, scalable, and maintainable foundation for a production-ready two-stage recommender system. The combination of modern design patterns, comprehensive ML pipeline, and production-focused considerations ensures the system can handle real-world loads while maintaining high quality recommendations.

The modular architecture allows for easy extension and modification of individual components without affecting the entire system. The use of industry-standard patterns and technologies ensures long-term maintainability and team productivity.

Key strengths of this architecture:
- **Scalability**: Horizontal and vertical scaling capabilities
- **Maintainability**: Clean separation of concerns and comprehensive documentation
- **Performance**: Optimized for high-throughput recommendation generation
- **Reliability**: Comprehensive error handling and monitoring
- **Flexibility**: Easy to add new features and ML models
- **Production-Ready**: Docker deployment, health checks, and monitoring

The system is designed to evolve with changing requirements and can incorporate new ML techniques and recommendation strategies as they become available.
