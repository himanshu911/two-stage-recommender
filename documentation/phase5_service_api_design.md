# Phase 5: Service Layer & API Design

## Overview: Connecting All the Pieces

In Phases 1-4, we learned:
- **Phase 1**: The problem (latency, scale, accuracy)
- **Phase 2**: The architecture (async, repository, service layer)
- **Phase 3**: The data (5 entities, feature store)
- **Phase 4**: The ML pipeline (two-stage, FAISS, logistic regression)

**Phase 5 answers**: How does it all connect? How does a user request flow through the system?

---

## The Complete Request Flow

```
HTTP Request
     │
     ▼
┌─────────────────────────────────────────────────┐
│  API Layer (recommendations.py)                  │
│  - Parse query params                            │
│  - Validate input                                │
│  - Handle errors → HTTP status codes             │
└──────────────────┬──────────────────────────────┘
                   │ Dependency Injection
                   ▼
┌─────────────────────────────────────────────────┐
│  RecommendationService (ORCHESTRATOR)            │
│  ├─ Check cache (10min TTL)                     │ 150ms
│  ├─ CandidateGenerationService (Stage 1)        │ (or 1ms if cached)
│  ├─ RankingService (Stage 2)                    │
│  ├─ UserRepository (fetch full user objects)    │
│  ├─ Build response                              │
│  ├─ Update cache                                │
│  └─ Track metrics                               │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│  Response (JSON)                                 │
│  {                                              │
│    "recommendations": [...],                    │
│    "total_count": 20,                          │
│    "algorithm_version": "v1",                  │
│    "generation_time_ms": 145.67                │
│  }                                              │
└─────────────────────────────────────────────────┘
```

---

## Service Layer: The Orchestrator Pattern

### RecommendationService - The Conductor

**Role**: Orchestrate the two-stage pipeline, manage caching, handle errors

```python
class RecommendationService:
    """Main service for generating recommendations."""

    def __init__(
        self,
        session: AsyncSession,
        feature_service: FeatureService,
        candidate_generation_service: CandidateGenerationService,
        ranking_service: RankingService
    ):
        # Dependencies injected by FastAPI
        self.feature_service = feature_service
        self.candidate_generation_service = candidate_generation_service
        self.ranking_service = ranking_service

        # Repositories
        self.user_repository = UserRepository(session)

        # In-memory cache
        self._recommendation_cache: Dict[str, Tuple[List[int], datetime]] = {}
        self._cache_ttl = timedelta(minutes=10)

        # Performance tracking
        self._performance_metrics: Dict[str, List[float]] = {
            "generation_time": [],
            "candidate_count": [],
            "cache_hit_rate": []
        }
```

**Why this structure?**
- ✅ **Dependency Injection**: Services passed in, not created internally (testable)
- ✅ **Single Responsibility**: Orchestration only, doesn't do ML itself
- ✅ **Stateful**: Manages cache and metrics (per-instance state)

---

### The Main Orchestration Method

```python
async def get_recommendations(
    self,
    user_id: int,
    limit: int = 20,
    exclude_seen: bool = True,
    filters: Optional[Dict[str, Any]] = None,
    use_cache: bool = True
) -> RecommendationResponse:
    """Orchestrate the two-stage recommendation pipeline."""

    start_time = time.time()

    # Step 1: Check Cache (Performance Optimization)
    cache_key = f"recs:{user_id}:{limit}:{filters}"
    if use_cache and cache_key in self._recommendation_cache:
        cached_user_ids, timestamp = self._recommendation_cache[cache_key]
        if datetime.utcnow() - timestamp < self._cache_ttl:
            # Cache hit! Return in ~1ms
            return await self._build_response_from_cache(user_id, cached_user_ids, start_time)

    # Step 2: Generate Candidates (Stage 1 of ML Pipeline)
    candidates = await self.candidate_generation_service.generate_candidates(
        user_id=user_id,
        limit=100,  # Configured: CANDIDATE_GENERATION_TOP_K
        exclude_seen=exclude_seen,
        filters=filters
    )

    if not candidates:
        # Graceful degradation: return empty response
        return self._empty_response(user_id, start_time)

    # Step 3: Rank Candidates (Stage 2 of ML Pipeline)
    ranked_candidates = await self.ranking_service.rank_candidates(
        user_id=user_id,
        candidate_ids=[c.user_id for c in candidates],
        limit=limit
    )

    if not ranked_candidates:
        # Graceful degradation: return empty response
        return self._empty_response(user_id, start_time)

    # Step 4: Fetch Full User Objects (from Repository Layer)
    recommended_user_ids = [rc.user_id for rc in ranked_candidates]
    recommended_users = await self.user_repository.get_all(
        limit=len(recommended_user_ids)
    )

    # Step 5: Build Response with Scores
    user_map = {user.id: user for user in recommended_users}
    score_map = {rc.user_id: rc.score for rc in ranked_candidates}

    response = RecommendationResponse(
        recommendations=[
            UserResponse(
                id=user.id,
                name=user.name,
                age=user.age,
                # ... other fields
                match_score=score_map.get(user.id, 0.0)  # Add ML score!
            )
            for user in recommended_users
        ],
        total_count=len(recommended_users),
        algorithm_version=self.ranking_service.model_version,
        generation_time_ms=(time.time() - start_time) * 1000
    )

    # Step 6: Update Cache
    if use_cache:
        self._recommendation_cache[cache_key] = (
            recommended_user_ids,
            datetime.utcnow()
        )

    # Step 7: Track Performance Metrics
    self._track_performance(len(recommended_users), generation_time_ms, use_cache)

    # Step 8: Log for Monitoring
    logger.info(
        "Recommendations generated",
        user_id=user_id,
        count=len(response.recommendations),
        generation_time_ms=generation_time_ms
    )

    return response
```

---

## Design Pattern Analysis

### Pattern 1: Orchestration (Service Coordination)

**What is Orchestration?**
- RecommendationService **coordinates** other services
- Doesn't do ML itself, delegates to specialized services
- Manages the workflow: cache → candidates → ranking → response

**Why Orchestration Pattern?**

**Without Orchestration** (API does everything):
```python
# Bad: API endpoint has all the logic
@app.get("/recommendations")
async def get_recommendations(user_id: int):
    # Cache logic
    if cache_has(user_id):
        return cache_get(user_id)

    # Stage 1
    cf_model = load_cf_model()
    candidates = cf_model.find_similar(user_id)

    # Stage 2
    ranking_model = load_ranking_model()
    scores = ranking_model.predict(candidates)

    # Build response
    users = db.query(User).filter(User.id.in_(candidates)).all()
    return users
```

**Problems**:
- ❌ API layer knows about ML models (tight coupling)
- ❌ Can't test orchestration without HTTP
- ❌ Can't reuse in other endpoints
- ❌ Hard to change workflow

**With Orchestration** (Service handles it):
```python
# Good: Service orchestrates
@app.get("/recommendations")
async def get_recommendations(
    user_id: int,
    rec_service: RecommendationServiceDep  # Injected!
):
    return await rec_service.get_recommendations(user_id)

# Orchestration logic in service
class RecommendationService:
    async def get_recommendations(self, user_id):
        # Check cache
        # Call CandidateGenerationService
        # Call RankingService
        # Build response
        return response
```

**Benefits**:
- ✅ API layer stays thin (just HTTP handling)
- ✅ Orchestration logic testable without HTTP
- ✅ Reusable in multiple endpoints
- ✅ Easy to modify workflow

---

### Pattern 2: Caching Strategy

**In-Memory Cache with TTL**:
```python
self._recommendation_cache: Dict[str, Tuple[List[int], datetime]] = {}
self._cache_ttl = timedelta(minutes=10)

# Cache key includes all parameters that affect results
cache_key = f"recs:{user_id}:{limit}:{str(sorted(filters.items()) if filters else '')}"

# Check cache with TTL
if cache_key in self._recommendation_cache:
    cached_user_ids, timestamp = self._recommendation_cache[cache_key]
    if datetime.utcnow() - timestamp < self._cache_ttl:
        return cached_result  # Cache hit!

# Cache miss: generate fresh recommendations
recommendations = await self._generate_fresh_recommendations()

# Store in cache
self._recommendation_cache[cache_key] = (
    recommendation_ids,
    datetime.utcnow()
)
```

**Key Design Decisions**:

#### Decision 1: Cache key includes all parameters
```python
# If different params, different cache entry
cache_key = f"recs:{user_id}:{limit}:{filters}"

# user_id=1, limit=20, no filters  → "recs:1:20:"
# user_id=1, limit=10, no filters  → "recs:1:10:"  (different key!)
# user_id=1, limit=20, min_age=25  → "recs:1:20:min_age=25"
```

**Why?** Different params = different recommendations. Must cache separately.

#### Decision 2: TTL = 10 minutes
```python
self._cache_ttl = timedelta(minutes=10)
```

**Trade-offs**:
| TTL | Pros | Cons |
|-----|------|------|
| **1 minute** | Very fresh | Low cache hit rate, more DB load |
| **10 minutes** (chosen) | Good balance | Recommendations can be stale |
| **1 hour** | High cache hit rate | Very stale, poor UX |

**Why 10 minutes?**
- ✅ Recommendations don't change that fast (user profiles stable)
- ✅ Good cache hit rate (~50-70%)
- ✅ Acceptable staleness (user won't notice)
- ❌ If user just liked someone, won't see immediate effect (acceptable trade-off)

#### Decision 3: Cache only user IDs, not full user objects
```python
# Store
self._recommendation_cache[cache_key] = (
    [1, 5, 10, 23, ...],  # Just IDs!
    timestamp
)

# Fetch full objects from DB when returning
users = await self.user_repository.get_by_ids(cached_ids)
```

**Why not cache full user objects?**
- ✅ **Smaller memory**: 100 IDs = 400 bytes, 100 User objects = 50KB
- ✅ **Fresh data**: User updates reflected immediately
- ❌ **Extra DB query**: Must fetch users from DB (but fast with index on ID)

**Trade-off**: Memory efficiency vs one extra DB query (acceptable)

#### Decision 4: Per-service instance cache (not shared)
```python
class RecommendationService:
    def __init__(self):
        # Each service instance has its own cache
        self._recommendation_cache: Dict = {}
```

**Current**: Each app server has separate cache
```
Server 1: Cache = {user_1: [...], user_5: [...]}
Server 2: Cache = {user_2: [...], user_8: [...]}
```

**Implications**:
- ✅ **Simple**: No external dependency (Redis)
- ✅ **Fast**: Memory access (~1ms)
- ❌ **Not shared**: Load balancer sends user_1 to Server 2 → cache miss
- ❌ **Lost on restart**: Server restart clears cache

**When to migrate to Redis?**
- Multiple servers (load balanced)
- High QPS (>10K requests/minute)
- Want to preserve cache across restarts

---

### Pattern 3: Error Handling & Graceful Degradation

**Philosophy**: System should work with reduced quality, not fail completely

```python
async def get_recommendations(self, user_id, limit):
    try:
        # Step 1: Generate candidates
        candidates = await self.candidate_generation_service.generate_candidates(...)

        if not candidates:
            # No candidates generated (maybe new user, no data)
            logger.warning("No candidates generated", user_id=user_id)
            return self._empty_response(user_id)  # Return empty, don't crash!

        # Step 2: Rank candidates
        ranked = await self.ranking_service.rank_candidates(...)

        if not ranked:
            # Ranking failed (maybe model not trained)
            logger.warning("No candidates ranked", user_id=user_id)
            return self._empty_response(user_id)  # Still don't crash!

        # Success path
        return response

    except Exception as e:
        # Unexpected error: log and return empty
        logger.error("Error generating recommendations", user_id=user_id, error=str(e))
        return self._empty_response(user_id, start_time)
```

**Levels of Degradation**:

1. **Best case**: Full ML pipeline (150ms, high quality)
2. **Degraded 1**: Ranking model fails → use fallback heuristics
3. **Degraded 2**: Candidate generation fails → use random users
4. **Degraded 3**: Everything fails → return empty list with error message

**Key Principle**: Never return 500 error if we can return something useful

---

### Pattern 4: Performance Tracking

```python
self._performance_metrics: Dict[str, List[float]] = {
    "generation_time": [],
    "candidate_count": [],
    "cache_hit_rate": []
}

def _track_performance(self, count, generation_time_ms, from_cache):
    self._performance_metrics["generation_time"].append(generation_time_ms)
    self._performance_metrics["candidate_count"].append(count)

    # Track cache hits
    if from_cache:
        cache_hits += 1

def get_performance_metrics(self) -> Dict[str, Any]:
    """Return performance statistics."""
    return {
        "avg_generation_time_ms": np.mean(self._performance_metrics["generation_time"]),
        "p95_generation_time_ms": np.percentile(self._performance_metrics["generation_time"], 95),
        "avg_candidate_count": np.mean(self._performance_metrics["candidate_count"]),
        "cache_hit_rate": cache_hits / total_requests
    }
```

**Why track metrics in-service?**
- ✅ **Real-time monitoring**: See performance without external tools
- ✅ **Debug performance issues**: Which step is slow?
- ✅ **Capacity planning**: When do we need more servers?

---

## API Layer: HTTP Interface

### API Design Principles

**RESTful Design**:
```
GET  /api/v1/recommendations/users/{user_id}/recommendations  # Get recs
GET  /api/v1/recommendations/users/{user_id}/similar          # Similar users
POST /api/v1/recommendations/refresh                          # Refresh cache
GET  /api/v1/recommendations/performance                      # Metrics
```

**Why RESTful?**
- ✅ Standard: Developers know how to use it
- ✅ Self-documenting: URL describes what it does
- ✅ HTTP verbs convey intent: GET (read), POST (write)

---

### API Endpoint Anatomy

```python
@router.get(
    "/users/{user_id}/recommendations",
    response_model=RecommendationResponse,
    summary="Get user recommendations",
    description="Get personalized recommendations for a user"
)
async def get_recommendations(
    user_id: int,                                # Path parameter
    recommendation_service: RecommendationServiceDep,  # Dependency injection
    limit: int = Query(20, ge=1, le=100),       # Query param with validation
    exclude_seen: bool = Query(True),           # Optional query param
    min_age: Optional[int] = Query(None, ge=18),  # Optional filter
    max_age: Optional[int] = Query(None, le=120),
    location: Optional[str] = Query(None)
) -> RecommendationResponse:
    """
    Get personalized recommendations for a user.
    """
    try:
        # Build filters dict from optional params
        filters = {}
        if min_age is not None:
            filters["min_age"] = min_age
        if max_age is not None:
            filters["max_age"] = max_age
        if location:
            filters["location"] = location

        # Call service (orchestration happens here)
        response = await recommendation_service.get_recommendations(
            user_id=user_id,
            limit=limit,
            exclude_seen=exclude_seen,
            filters=filters
        )

        # Log for monitoring
        logger.info(
            "Recommendations generated",
            user_id=user_id,
            count=len(response.recommendations)
        )

        return response

    except Exception as e:
        # Error handling: service layer errors → HTTP errors
        logger.error("Error generating recommendations", user_id=user_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate recommendations"
        )
```

**Key API Design Decisions**:

#### Decision 1: Query parameters for filters (not body)
```
# Good: Query params for GET request
GET /recommendations/users/1/recommendations?limit=20&min_age=25&max_age=35

# Bad: Request body for GET (non-standard)
GET /recommendations/users/1/recommendations
Body: {"limit": 20, "min_age": 25}
```

**Why query params?**
- ✅ RESTful convention: GET requests don't have bodies
- ✅ Cacheable: Can cache based on URL
- ✅ Bookmarkable: Can save URL

#### Decision 2: FastAPI Query validation
```python
limit: int = Query(20, ge=1, le=100, description="Number of recommendations")
```

**What this does**:
- **Default**: If not provided, use 20
- **Validation**: Must be >= 1 and <= 100
- **Auto docs**: Description appears in Swagger UI
- **Type checking**: Must be int, not string

**Before request reaches service**:
```
Request: /recommendations?limit=200  → Error 422 (validation failed)
Request: /recommendations?limit=abc  → Error 422 (not an integer)
Request: /recommendations?limit=20   → ✓ Passes to service
```

#### Decision 3: Dependency injection for services
```python
async def get_recommendations(
    user_id: int,
    recommendation_service: RecommendationServiceDep  # Injected by FastAPI!
):
    return await recommendation_service.get_recommendations(user_id)
```

**How dependency injection works** (from Phase 2):
```python
# FastAPI automatically:
# 1. Calls get_recommendation_service()
# 2. That function creates/fetches RecommendationService
# 3. Injects it into endpoint function
# 4. Cleans up after request completes
```

**Benefits**:
- ✅ API doesn't create services (loose coupling)
- ✅ Easy to test (mock the service)
- ✅ Service lifecycle managed automatically

#### Decision 4: Structured error responses
```python
try:
    return await recommendation_service.get_recommendations(...)
except Exception as e:
    logger.error("Error generating recommendations", error=str(e))
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Failed to generate recommendations"
    )
```

**Error response format**:
```json
{
  "error": "Failed to generate recommendations",
  "detail": "...",
  "request_id": "uuid-12345"
}
```

**Why structured errors?**
- ✅ **Client-friendly**: Clear error message
- ✅ **Debuggable**: request_id for log correlation
- ✅ **Standard**: HTTP status codes convey error type

---

### API Response Design

```python
class RecommendationResponse(BaseModel):
    """Response model for recommendations."""
    recommendations: List[UserResponse]
    total_count: int
    algorithm_version: str
    generation_time_ms: float

class UserResponse(BaseModel):
    """User info in recommendation response."""
    id: int
    name: str
    age: int
    gender: str
    location: str
    bio: Optional[str]
    interests: Optional[List[str]]
    match_score: float  # ML score!
    created_at: datetime
    last_active_at: Optional[datetime]
```

**Example API Response**:
```json
{
  "recommendations": [
    {
      "id": 42,
      "name": "Alice",
      "age": 28,
      "gender": "female",
      "location": "San Francisco",
      "bio": "Love hiking and photography",
      "interests": ["hiking", "photography", "travel"],
      "match_score": 0.89,
      "created_at": "2024-01-15T10:30:00Z",
      "last_active_at": "2024-01-20T15:45:00Z"
    },
    // ... 19 more users
  ],
  "total_count": 20,
  "algorithm_version": "v1.0.0",
  "generation_time_ms": 145.67
}
```

**Key Design Decisions**:

#### Decision 1: Include match_score in response
```python
match_score: float  # Probability from ranking model (0-1)
```

**Why expose ML score to client?**
- ✅ **Transparency**: User sees why they were matched
- ✅ **Sorting**: Client can re-sort by score if needed
- ✅ **A/B testing**: Can compare scores across algorithms
- ❌ **Privacy concern**: Reveals internal scoring (acceptable trade-off)

#### Decision 2: Include metadata (algorithm_version, generation_time_ms)
```python
algorithm_version: str          # "v1.0.0"
generation_time_ms: float       # 145.67
```

**Why include metadata?**
- ✅ **A/B testing**: Client knows which algorithm served this
- ✅ **Performance monitoring**: Client can track latency
- ✅ **Debugging**: Can reproduce issues ("this happened on v2.0")

#### Decision 3: Pagination via limit parameter (not offset)
```python
limit: int = Query(20, ge=1, le=100)  # Just limit, no offset
```

**Why no offset?**
- Recommendations are personalized and change over time
- offset=20 might return same users as offset=0 (if recs changed)
- Better: "Load more" fetches fresh recommendations each time

**Alternative approaches**:
- **Cursor-based pagination**: Return next_cursor token
- **Infinite scroll**: Load more on scroll (uses limit only)
- **Full pagination**: offset + limit (for static lists only)

---

## Complete Request Flow (End-to-End)

Let's trace a real request through the entire system:

### Request
```http
GET /api/v1/recommendations/users/123/recommendations?limit=20&min_age=25&max_age=35
```

### Step-by-Step Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. FastAPI receives HTTP request                            │
│    - Parses URL: user_id=123                               │
│    - Parses query params: limit=20, min_age=25, max_age=35 │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. FastAPI validates parameters                             │
│    - limit=20: ✓ (between 1 and 100)                       │
│    - min_age=25: ✓ (>= 18)                                 │
│    - max_age=35: ✓ (<= 120)                                │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. FastAPI calls dependency injection chain                 │
│    get_recommendation_service()                             │
│      → creates RecommendationService                        │
│      → injects CandidateGenerationService                  │
│      → injects RankingService                              │
│      → injects FeatureService                              │
│      → injects AsyncSession (database)                     │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. API endpoint calls service                               │
│    response = await recommendation_service.get_recommendations(
│        user_id=123,
│        limit=20,
│        filters={"min_age": 25, "max_age": 35}
│    )
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. RecommendationService: Check cache                       │
│    cache_key = "recs:123:20:min_age=25,max_age=35"        │
│    if cache_key in cache and not expired:                  │
│        return cached_result  → DONE (1ms)                  │
│    else:                                                    │
│        continue to pipeline                                 │
└──────────────────┬──────────────────────────────────────────┘
                   │ Cache Miss
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. Stage 1: CandidateGenerationService                      │
│    candidates = await candidate_service.generate_candidates(│
│        user_id=123,                                         │
│        limit=100,                                           │
│        filters={"min_age": 25, "max_age": 35}             │
│    )                                                        │
│    → Returns 100 candidate user IDs                        │
│    Time: ~100ms                                             │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ 7. Stage 2: RankingService                                  │
│    ranked = await ranking_service.rank_candidates(          │
│        user_id=123,                                         │
│        candidate_ids=[45, 67, 89, ...],  # 100 IDs        │
│        limit=20                                             │
│    )                                                        │
│    → Returns top 20 candidates with ML scores              │
│    Time: ~50ms                                              │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ 8. UserRepository: Fetch full user objects                  │
│    users = await user_repository.get_by_ids([45, 67, ...])│
│    Time: ~10ms (indexed query)                              │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ 9. Build response                                           │
│    response = RecommendationResponse(                       │
│        recommendations=[...],                               │
│        total_count=20,                                      │
│        algorithm_version="v1",                              │
│        generation_time_ms=160.5                             │
│    )                                                        │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ 10. Cache the result                                        │
│     cache["recs:123:20:min_age=25,max_age=35"] = (         │
│         [45, 67, 89, ...],  # Just IDs                    │
│         datetime.now()                                      │
│     )                                                       │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ 11. Track metrics                                           │
│     metrics["generation_time"].append(160.5)               │
│     metrics["candidate_count"].append(20)                  │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ 12. Log for monitoring                                      │
│     logger.info(                                            │
│         "Recommendations generated",                        │
│         user_id=123,                                        │
│         count=20,                                           │
│         generation_time_ms=160.5                            │
│     )                                                       │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ 13. FastAPI serializes response to JSON                     │
│     {                                                       │
│       "recommendations": [...],                             │
│       "total_count": 20,                                    │
│       "algorithm_version": "v1",                            │
│       "generation_time_ms": 160.5                           │
│     }                                                       │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ 14. HTTP Response sent to client                            │
│     Status: 200 OK                                          │
│     Headers:                                                │
│       Content-Type: application/json                        │
│       X-Request-ID: uuid-12345                             │
│       X-Process-Time: 0.165                                 │
└─────────────────────────────────────────────────────────────┘

Total Time: ~165ms (within 300ms budget ✓)
```

---

## Design Patterns Summary

| Pattern | Purpose | Implementation | Benefits |
|---------|---------|----------------|----------|
| **Orchestration** | Coordinate services | RecommendationService calls CandidateGen + Ranking + Repository | Clean separation, testable, reusable |
| **Caching** | Reduce latency | In-memory dict with TTL | 100x faster (1ms vs 150ms) |
| **Graceful Degradation** | Handle failures | Return empty/fallback on errors | System works with reduced quality |
| **Dependency Injection** | Manage dependencies | FastAPI Depends() | Loose coupling, easy testing |
| **Performance Tracking** | Monitor system | In-service metrics collection | Real-time performance visibility |
| **Error Handling** | HTTP error responses | try/except → HTTPException | User-friendly errors, debuggable |

---

## Interview Narratives

### Q: "Walk me through how a recommendation request flows through your system"

**Strong Answer (90 seconds)**:

> "When a user requests recommendations, it flows through three layers.
>
> **API Layer** receives the HTTP GET request, validates query parameters using FastAPI's Query validation - for example, limit must be between 1 and 100. Then FastAPI's dependency injection creates all the required services - RecommendationService, which depends on CandidateGenerationService, RankingService, and FeatureService.
>
> **Service Layer** - RecommendationService acts as the orchestrator. First, it checks an in-memory cache with a 10-minute TTL. If cache hit, we return in 1ms. On cache miss, we run the two-stage pipeline: CandidateGenerationService generates 100 candidates using collaborative filtering, content-based, and random exploration - this takes about 100ms. Then RankingService scores those 100 candidates with our logistic regression model - about 50ms. Finally, we fetch full user objects from the database using UserRepository.
>
> **Response Building** - We construct the API response with user details, ML scores, metadata like algorithm version and generation time. We cache the result (just user IDs for memory efficiency), track performance metrics, and log for monitoring.
>
> Total time: 165ms on cache miss, 1ms on cache hit - well within our 300ms budget."

---

### Q: "How does your caching strategy work?"

**Strong Answer (45 seconds)**:

> "We use in-memory caching with a 10-minute TTL. The cache key includes user ID, limit, and all filters - different parameters create different cache entries. We cache only user IDs, not full user objects, for memory efficiency - 100 IDs is 400 bytes versus 50KB for full objects.
>
> Cache hit rate is typically 50-70%, which is huge for performance - going from 165ms to 1ms. The TTL of 10 minutes balances freshness and hit rate - recommendations don't change that fast, so staleness is acceptable.
>
> Currently it's per-instance in-memory, which works for a single server. When we scale to multiple servers, we'll migrate to Redis for a shared cache. But for now, in-memory gives us microsecond access with zero external dependencies."

---

### Q: "How do you handle errors in the service layer?"

**Strong Answer (45 seconds)**:

> "We follow graceful degradation - the system works with reduced quality rather than failing completely.
>
> For example, if candidate generation returns no candidates - maybe a new user with no data - we don't crash. We log a warning and return an empty response with a clear message. If the ranking model isn't trained yet, we have a fallback ranking using simple heuristics like profile completeness and recent activity.
>
> At the API layer, we catch exceptions and translate them to HTTP errors. We log the error with context like user_id and request_id, then return a structured error response with appropriate status code. This way, the client gets a clear error message, and we have logs to debug the issue.
>
> The principle is: never return 500 if we can return something useful, even if it's just an empty list."

---

### Q: "Why does RecommendationService orchestrate other services rather than doing everything itself?"

**Strong Answer (45 seconds)**:

> "Single Responsibility Principle. RecommendationService's job is orchestration - managing the workflow, caching, and error handling. It delegates the actual ML work to specialized services.
>
> This has three key benefits. First, testability - we can mock CandidateGenerationService and test orchestration logic independently. Second, reusability - if we want similar users in another endpoint, we can reuse CandidateGenerationService. Third, maintainability - if we upgrade the ranking model, we only change RankingService, not the orchestration logic.
>
> It's the same reason conductors don't play instruments - they coordinate musicians. RecommendationService coordinates ML services."

---

### Q: "How would you add a new recommendation algorithm?"

**Strong Answer (60 seconds)**:

> "We'd leverage our existing architecture with minimal changes.
>
> **Step 1**: Create a new RankingService with the new model - let's call it RankingServiceV2. It implements the same interface: `rank_candidates()` returns RankedCandidate objects.
>
> **Step 2**: Update dependency injection to support algorithm versioning. The API endpoint already accepts an `algorithm_version` parameter. Based on that parameter, we'd inject either RankingService or RankingServiceV2.
>
> **Step 3**: A/B testing - route 50% of requests to v1, 50% to v2. Track metrics for each version separately. The algorithm_version field in the response tells us which model served each request.
>
> **Step 4**: Gradual rollout - if v2 performs better, gradually shift more traffic. If worse, roll back instantly by changing the active version.
>
> The key is our service-oriented architecture makes it easy to swap implementations without touching the API or orchestration layer."

---

## Summary: Service Layer Connects Everything

| Component (Previous Phases) | How Service Layer Uses It | Why This Matters |
|-----------------------------|---------------------------|------------------|
| **Repository Pattern** (Phase 2) | UserRepository.get_by_ids() | Fetches full user objects after ranking |
| **Feature Store** (Phase 3) | FeatureService.get_features() | Provides features for ranking model |
| **Two-Stage Pipeline** (Phase 4) | CandidateGen + Ranking services | Orchestrates the complete ML flow |
| **Caching** (Phase 3) | In-memory cache with TTL | 100x speedup on cache hit |
| **Error Handling** (Phase 2) | Graceful degradation | System works with reduced quality |

---

## Next Steps
- [ ] Validate understanding of service orchestration
- [ ] Move to Phase 6: Production Considerations (monitoring, scaling, deployment)
- [ ] Or skip to Phase 7: Interview Narrative Synthesis