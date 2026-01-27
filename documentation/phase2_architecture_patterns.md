# Phase 2: High-Level System Design & Architecture Patterns

## Overview: How Architecture Addresses Requirements

In Phase 1, we identified key requirements:
- **Latency**: <300ms p95
- **Scale**: 100K-1M concurrent users
- **Accuracy**: High match rate
- **Freshness**: Real-time feedback incorporation
- **Maintainability**: Testable, modular code

**This phase explains HOW the architecture addresses these requirements.**

---

## Architecture Decision #1: FastAPI + Async/Await

### The Decision
```python
# app/main.py
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    await create_tables()  # Async!
    yield

# All endpoints are async
async def get_recommendations(...):
    recommendations = await recommendation_service.get_recommendations(user_id)
```

### Why FastAPI?

**Problem**: Need high throughput (10K-100K requests/min) with low latency

**Alternatives Considered**:
| Framework | Pros | Cons | Why Not? |
|-----------|------|------|----------|
| Flask | Simple, mature | Sync-only, slower | Can't handle 10K concurrent I/O-bound requests efficiently |
| Django | Batteries included | Heavy, sync by default | Too much overhead for API-only service |
| FastAPI | Async, fast, modern | Newer ecosystem | ✓ **Chosen**: Perfect for async I/O-bound workloads |

**Key Benefits**:
1. **Async/await** = Non-blocking I/O
   - While waiting for database: handle other requests
   - **10x more concurrent requests** with same hardware
   - Critical for latency budget (300ms)

2. **Automatic API documentation** (OpenAPI/Swagger)
   - `/docs` endpoint for free
   - Client SDK generation
   - API contract validation

3. **Type hints + Pydantic validation**
   - Request validation automatic
   - Response serialization built-in
   - Catches errors before DB hits

### Trade-offs
| Pro | Con |
|-----|-----|
| High performance for I/O | Requires async mindset |
| Auto docs | Newer, smaller ecosystem |
| Type safety | Must use async libraries |

### Interview Answer (30 seconds)
> "We chose FastAPI with async/await because we're I/O-bound - we spend most time waiting on database, feature computation, and ML inference. Async lets us handle thousands of concurrent requests without blocking. For example, while one request waits for the database, we can process 10 others. This directly addresses our 300ms latency target and 100K concurrent user requirement."

---

## Architecture Decision #2: Repository Pattern

### The Decision
```python
# app/repositories/base.py
class BaseRepository(Generic[T], ABC):
    @abstractmethod
    async def get_by_id(self, id: int) -> Optional[T]: pass

    @abstractmethod
    async def create(self, entity: T) -> T: pass

# app/repositories/user_repository.py
class UserRepository(SQLModelRepository[User]):
    async def get_by_location(self, location: str) -> List[User]:
        # Domain-specific query
```

### Why Repository Pattern?

**Problem**: API layer shouldn't know about database implementation details

**Without Repository Pattern**:
```python
# Bad: API directly uses SQLAlchemy
@app.get("/users/{user_id}")
async def get_user(user_id: int, session: AsyncSession = Depends(get_session)):
    # API knows about SELECT, WHERE, etc.
    result = await session.execute(select(User).where(User.id == user_id))
    return result.scalar_one_or_none()
```

**Problems with above**:
- API layer coupled to SQLAlchemy
- Can't test without database
- Can't switch databases easily
- Duplicate query logic everywhere

**With Repository Pattern**:
```python
# Good: API uses repository abstraction
@app.get("/users/{user_id}")
async def get_user(user_id: int, user_repo: UserRepositoryDep):
    # API doesn't know about database
    return await user_repo.get_by_id(user_id)
```

### Benefits

#### 1. Testability
```python
# Easy to mock in tests
class MockUserRepository:
    async def get_by_id(self, id: int):
        return User(id=id, name="Test User")

# Test doesn't need real database!
```

#### 2. Abstraction
- API layer doesn't know if data comes from PostgreSQL, MongoDB, or Redis
- Can switch databases without changing API code
- Single place to optimize queries

#### 3. Domain Logic Centralization
```python
# Domain-specific queries live in repository
class UserRepository:
    async def get_active_users_in_location(
        self, location: str, last_active_days: int = 7
    ) -> List[User]:
        # Complex query encapsulated
        query = select(User).where(
            User.location == location,
            User.last_active_at > datetime.now() - timedelta(days=last_active_days)
        )
        ...
```

### Trade-offs
| Pro | Con |
|-----|-----|
| Testable without DB | Extra layer of abstraction |
| Swappable data source | More boilerplate code |
| Centralized query logic | Learning curve |

### Interview Answer (30 seconds)
> "We use the repository pattern to abstract data access from business logic. This gives us three key benefits: First, testability - we can mock repositories without a real database. Second, we can swap storage backends without touching API code. Third, domain-specific queries live in one place. For example, UserRepository.get_active_users_in_location() encapsulates the complex query logic."

---

## Architecture Decision #3: Service Layer

### The Decision
```python
# app/services/recommendation_service.py
class RecommendationService:
    def __init__(
        self,
        session: AsyncSession,
        feature_service: FeatureService,
        candidate_generation_service: CandidateGenerationService,
        ranking_service: RankingService
    ):
        # Service orchestrates other services
        self.candidate_generation_service = candidate_generation_service
        self.ranking_service = ranking_service

    async def get_recommendations(self, user_id: int, limit: int) -> RecommendationResponse:
        # Step 1: Generate candidates
        candidates = await self.candidate_generation_service.generate_candidates(user_id)

        # Step 2: Rank candidates
        ranked = await self.ranking_service.rank_candidates(user_id, candidates)

        # Step 3: Build response
        return self._build_response(ranked)
```

### Why Service Layer?

**Problem**: Business logic shouldn't live in API endpoints or repositories

**Three-Layer Architecture**:
```
┌─────────────────────────────────────┐
│      API Layer (FastAPI)            │  ← HTTP handling, validation
├─────────────────────────────────────┤
│      Service Layer                  │  ← Business logic, orchestration
│  - RecommendationService            │
│  - CandidateGenerationService       │
│  - RankingService                   │
│  - FeatureService                   │
├─────────────────────────────────────┤
│      Repository Layer               │  ← Data access
│  - UserRepository                   │
│  - InteractionRepository            │
└─────────────────────────────────────┘
```

### Responsibilities by Layer

#### API Layer (app/api/v1/)
- HTTP request/response handling
- Input validation (Pydantic models)
- Authentication/authorization (future)
- Error handling (HTTP status codes)
- **No business logic!**

#### Service Layer (app/services/)
- **Business logic**: "What is a recommendation?"
- **Orchestration**: Coordinate multiple repositories/services
- **Transaction boundaries**: When to commit/rollback
- **Caching**: Performance optimization
- **Logging/metrics**: Observability

#### Repository Layer (app/repositories/)
- **Data access only**: CRUD operations
- **Query building**: SQL generation
- **ORM mapping**: Database ↔ Model conversion
- **No business logic!**

### Example: Two-Stage Recommendation

```python
# Service orchestrates the pipeline
class RecommendationService:
    async def get_recommendations(self, user_id: int, limit: int):
        """Business logic: two-stage recommendation."""
        start_time = time.time()

        # Check cache (business logic)
        if use_cache:
            cached = self._get_from_cache(user_id)
            if cached:
                return cached

        # Stage 1: Generate candidates (orchestration)
        candidates = await self.candidate_generation_service.generate_candidates(
            user_id,
            limit=100  # Business decision: retrieve 100, rank to 20
        )

        # Stage 2: Rank candidates (orchestration)
        ranked = await self.ranking_service.rank_candidates(user_id, candidates)

        # Apply filters (business logic)
        filtered = self._apply_business_filters(ranked, user_preferences)

        # Update cache (business logic)
        self._update_cache(user_id, filtered)

        # Log metrics (observability)
        logger.info("Recommendations generated",
                   user_id=user_id,
                   count=len(filtered),
                   generation_time_ms=time.time() - start_time)

        return self._build_response(filtered)
```

### Benefits

#### 1. Single Responsibility
Each service has one job:
- `CandidateGenerationService`: Generate candidates only
- `RankingService`: Rank candidates only
- `FeatureService`: Extract features only
- `RecommendationService`: Orchestrate the pipeline

#### 2. Testability
```python
# Mock services, not databases
class MockCandidateGenerationService:
    async def generate_candidates(self, user_id: int):
        return [Candidate(id=1), Candidate(id=2)]

# Test recommendation logic without real candidate generation
recommendation_service = RecommendationService(
    candidate_generation_service=MockCandidateGenerationService()
)
```

#### 3. Reusability
```python
# Same service used by different endpoints
@app.get("/recommendations")
async def get_recs(rec_service: RecommendationServiceDep):
    return await rec_service.get_recommendations(user_id)

@app.get("/recommendations/explain")
async def explain_recs(rec_service: RecommendationServiceDep):
    return await rec_service.get_recommendations_with_explanation(user_id)
```

### Trade-offs
| Pro | Con |
|-----|-----|
| Clear boundaries | More files/classes |
| Easy to test | Dependency chain complexity |
| Reusable logic | Potential over-engineering |

### Interview Answer (45 seconds)
> "We separate concerns into three layers. The API layer handles HTTP - validation, errors, status codes. The service layer contains all business logic - what is a recommendation, how do we generate it, when do we cache. The repository layer handles data access only.
>
> For example, RecommendationService orchestrates our two-stage pipeline: it calls CandidateGenerationService to get 100 candidates, then RankingService to score them down to 20. It also handles caching, logging, and metrics. This separation makes each component testable in isolation."

---

## Architecture Decision #4: Dependency Injection

### The Decision
```python
# app/core/dependencies.py
async def get_recommendation_service(
    session: SessionDep,
    feature_service: Annotated[FeatureService, Depends(get_feature_service)],
    candidate_service: Annotated[CandidateGenerationService, Depends(get_candidate_generation_service)],
    ranking_service: Annotated[RankingService, Depends(get_ranking_service)]
) -> RecommendationService:
    """Dependency injection: FastAPI creates and wires services."""
    return RecommendationService(
        session=session,
        feature_service=feature_service,
        candidate_generation_service=candidate_service,
        ranking_service=ranking_service
    )

# API endpoint just declares what it needs
@app.get("/recommendations")
async def get_recs(
    user_id: int,
    rec_service: RecommendationServiceDep  # FastAPI injects this!
):
    return await rec_service.get_recommendations(user_id)
```

### Why Dependency Injection?

**Problem**: Services have complex dependency trees

```
RecommendationService
├── FeatureService
│   └── AsyncSession
├── CandidateGenerationService
│   ├── FeatureService
│   │   └── AsyncSession
│   └── AsyncSession
└── RankingService
    ├── FeatureService
    │   └── AsyncSession
    └── AsyncSession
```

**Without DI** (manual wiring):
```python
# Bad: Manual dependency management
@app.get("/recommendations")
async def get_recs(user_id: int):
    # Create session
    session = get_session()

    # Create feature service
    feature_service = FeatureService(session)

    # Create candidate service
    candidate_service = CandidateGenerationService(session, feature_service)

    # Create ranking service
    ranking_service = RankingService(session, feature_service)

    # Finally create recommendation service
    rec_service = RecommendationService(
        session, feature_service, candidate_service, ranking_service
    )

    # Now we can use it!
    return await rec_service.get_recommendations(user_id)
```

**Problems**:
- Lots of boilerplate
- Hard to test (can't inject mocks)
- Must remember dependency order
- Duplicate wiring in every endpoint

**With DI** (FastAPI manages it):
```python
# Good: FastAPI handles wiring
@app.get("/recommendations")
async def get_recs(
    user_id: int,
    rec_service: RecommendationServiceDep  # Just declare what you need!
):
    return await rec_service.get_recommendations(user_id)
```

### Benefits

#### 1. Testability
```python
# Easy to inject mocks
def test_get_recommendations():
    # Create mock service
    mock_service = MockRecommendationService()

    # Override dependency
    app.dependency_overrides[get_recommendation_service] = lambda: mock_service

    # Test endpoint with mock
    response = client.get("/recommendations?user_id=1")
    assert response.status_code == 200
```

#### 2. Lifecycle Management
```python
# Session is created per-request and cleaned up automatically
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session  # FastAPI calls this
        # Automatic cleanup after request
```

#### 3. Configuration-based Behavior
```python
# Different implementations for dev/prod
if settings.ENVIRONMENT == "production":
    cache_service = RedisCache()
else:
    cache_service = InMemoryCache()
```

### Dependency Chain Example

```python
# FastAPI automatically resolves this chain:

# 1. User makes request
GET /recommendations?user_id=1

# 2. FastAPI sees: needs RecommendationServiceDep
#    Calls: get_recommendation_service()

# 3. get_recommendation_service needs:
#    - SessionDep → calls get_session()
#    - FeatureServiceDep → calls get_feature_service()
#    - CandidateServiceDep → calls get_candidate_generation_service()
#    - RankingServiceDep → calls get_ranking_service()

# 4. get_feature_service needs:
#    - SessionDep → reuses session from step 3!

# 5. All dependencies resolved, creates RecommendationService
#    Injects into endpoint

# 6. After response, FastAPI cleans up:
#    - Closes database session
#    - Cleans up resources
```

### Trade-offs
| Pro | Con |
|-----|-----|
| Loose coupling | "Magic" behavior (implicit) |
| Easy testing | Must understand DI pattern |
| Automatic cleanup | Debug complexity |

### Interview Answer (30 seconds)
> "We use FastAPI's dependency injection to manage service lifecycles. Instead of manually creating services in every endpoint, we declare what we need and FastAPI wires it up. For example, RecommendationService depends on FeatureService, CandidateService, and RankingService - FastAPI creates all of them in the right order and injects them. This makes testing easy - we can override dependencies with mocks."

---

## Architecture Summary: Requirements → Decisions

| Requirement | Architectural Decision | How It Helps |
|-------------|----------------------|--------------|
| **Latency <300ms** | FastAPI + Async/await | Non-blocking I/O, handle 1000s of concurrent requests |
| **Scale (100K users)** | Async architecture | 10x more concurrent requests per server |
| **Testability** | Repository pattern + DI | Mock dependencies, test without DB |
| **Maintainability** | Service layer | Clear boundaries, single responsibility |
| **Flexibility** | Repository abstraction | Swap databases, add features without changing API |
| **Observability** | Service layer logging | Centralized metrics, monitoring |
| **Development velocity** | FastAPI auto-docs | API contract clarity, client SDK generation |

---

## Interview Narrative: "Tell me about your architecture"

**Opening (30 seconds)**:
> "We use a three-layer architecture: API, Service, and Repository layers. The API layer is FastAPI with async/await for high concurrency - we handle 10K+ requests per minute. The service layer contains all business logic, like our two-stage recommendation pipeline. The repository layer abstracts data access for testability."

**Deep Dive (if asked "Why async?")**:
> "We're I/O-bound - spend most time waiting on database and ML inference. Async lets us handle thousands of concurrent requests without blocking. While one request waits for the database, we process 10 others. This directly addresses our 300ms latency target."

**Deep Dive (if asked "Why repository pattern?")**:
> "Testability and abstraction. We can test API endpoints by mocking repositories - no database needed. Also, if we switch from PostgreSQL to another database, only the repository layer changes, not the API or business logic."

**Deep Dive (if asked "Why service layer?")**:
> "Separation of concerns. The API layer handles HTTP, the service layer handles business logic like 'what is a recommendation', and the repository handles data access. For example, RecommendationService orchestrates our two-stage pipeline by calling CandidateGenerationService and RankingService."

---

## Common Follow-up Questions

### Q: "Why not use a monorepo with microservices?"

**Short Answer (30 seconds)**:
> "We started with a modular monolith because we're a small team. Our service layer already has clean boundaries - we could split into microservices later if needed. But premature microservices add operational complexity (deployment, monitoring, distributed transactions) without benefit at our scale."

**Deep Dive - The Real-World Split** (Conway's Law):

In practice, microservices are often driven by **team boundaries**, not just technical scaling.

**Typical Pattern**:
```
Monolith (One Team)
┌─────────────────────────────┐
│  Backend + ML (All Together)│
└─────────────────────────────┘
      ↓ Company grows
      ↓ Backend team + ML team form

Microservices (Two Teams)
┌──────────────┐  ┌──────────────┐
│Backend Service│  │  ML Service  │
│(Platform Team)│  │  (ML Team)   │
│- User CRUD   │  │- Recs        │
│- Interactions│  │- Models      │
└──────────────┘  └──────────────┘
```

**When we'd split**:
1. **Team autonomy**: Backend team wants to deploy without waiting for ML team
2. **Different release cadence**: Backend deploys weekly, ML deploys when models improve
3. **Clear ownership**: Who's on-call for recommendation bugs? ML team.
4. **Resource isolation**: ML inference needs GPU servers, Backend doesn't

**Conway's Law in action**: If you have 2 teams (Backend + ML), you'll naturally want 2 services.

**Current decision**: We're one team (or working closely), so monolith is appropriate. If we grow to separate Backend and ML teams with different priorities, we'd split:
- **Backend Service**: User management, interaction logging (owned by Platform team)
- **ML Service**: Recommendation generation, model training (owned by ML team)
- **API contract**: ML service calls Backend API to get interactions for training

This is the most common reason for microservices in ML organizations - team boundaries, not technical scaling.

### Q: "How do you handle database connections?"
**Answer**: "We use SQLAlchemy's async connection pool with dependency injection. Each request gets a session from the pool, uses it, and returns it automatically. FastAPI's DI handles lifecycle - session is created on request start, committed/rolled back on success/error, and closed after response."

### Q: "What if a service call fails?"
**Answer**: "We have error handling at each layer. Services catch exceptions, log them with context, and return appropriate errors. API layer translates to HTTP status codes. For example, if candidate generation fails, we log the error, and the API returns 500 with a request ID for debugging."

---

## Next Steps
- [ ] Validate understanding by reading actual code in app/api/, app/services/, app/repositories/
- [ ] Move to Phase 3: Deep dive into data models and storage layer
- [ ] Document how these patterns enable the ML pipeline (Phase 4)
