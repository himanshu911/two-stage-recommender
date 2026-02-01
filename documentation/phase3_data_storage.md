# Phase 3: Data Model & Storage Layer Design

## Overview: From Requirements to Schema Design

In Phase 1, we identified what data we need. In Phase 3, we answer: **How do we store and organize that data?**

**Key Design Questions**:
1. What entities do we need?
2. Why PostgreSQL for everything?
3. How do we store ML artifacts (embeddings, models)?
4. How do we ensure data consistency?
5. How do we make queries fast?

---

## The Five Core Entities

### Entity Relationship Diagram

```
┌─────────────┐
│    User     │
│             │
│ - id        │
│ - name      │
│ - age       │
│ - gender    │
│ - location  │
│ - interests │
└──────┬──────┘
       │
       │ 1:N interactions_made
       ↓
┌──────────────┐
│ Interaction  │
│              │
│ - user_id    │───────────┐
│ - target_id  │           │ N:1
│ - type       │           ↓
│ - timestamp  │      ┌─────────────┐
└──────────────┘      │    User     │
                      │  (target)   │
                      └─────────────┘

User 1:1 UserEmbedding (ML features)
User 1:N UserFeatures (versioned features)
MLModel (standalone, tracks model versions)
```

---

## Entity #1: User (Core Domain Entity)

### Schema Design

```python
class User(SQLModel, table=True):
    # Identity
    id: Optional[int]                    # Primary key
    name: str                           # Display name
    age: int                            # For filtering/matching
    gender: str                         # For filtering
    location: str                       # For geo-based matching

    # Profile content
    bio: Optional[str]                  # Text field for content-based features
    interests: Optional[List[str]]      # JSON array for matching

    # Timestamps (critical for production)
    created_at: datetime                # Account age feature
    updated_at: datetime                # Data freshness
    last_active_at: Optional[datetime]  # Activity-based filtering
```

### Design Decisions

#### Decision 1: JSON for interests array
```python
interests: Optional[List[str]] = Field(sa_column=Column(JSON))
```

**Why JSON instead of separate table?**

**Alternative 1: Normalized (Separate Table)**
```sql
-- users table
id | name | age

-- interests table
id | name

-- user_interests junction table
user_id | interest_id
```

**Alternative 2: JSON Array (Chosen)**
```sql
-- users table
id | name | age | interests
1  | Alice| 25  | ["hiking", "photography"]
```

**Trade-offs**:
| Aspect | Normalized (3 tables) | JSON Array (1 table) |
|--------|----------------------|---------------------|
| **Queries** | JOINs required | Single table lookup |
| **Inserts** | 3 tables to update | 1 table to update |
| **Filtering** | Easy with SQL WHERE | Use JSON operators |
| **Flexibility** | Fixed schema | Add new interests anytime |
| **Performance** | Slower (JOINs) | Faster (single table) |

**Why we chose JSON**:
- ✅ **Faster reads**: Single table, no JOINs (critical for 300ms latency)
- ✅ **Flexible schema**: Can add any interest without migrations
- ✅ **PostgreSQL JSON support**: Can query with `interests @> '["hiking"]'`
- ✅ **Simpler code**: No junction table management

**When normalized would be better**:
- If we needed interest statistics across all users
- If we had 100+ interests per user (rare in dating)
- If we needed referential integrity on interests

#### Decision 2: Timezone-aware timestamps

```python
created_at: datetime = Field(
    default_factory=lambda: datetime.now(timezone.utc),
    sa_column=Column(DateTime(timezone=True), server_default=func.now())
)
```

**Why timezone-aware?**
- Users are global → different timezones
- Prevents bugs: "User created in the future"
- Best practice: Always store UTC, convert to local for display

**Why server_default=func.now()?**
- Database sets timestamp, not Python
- Consistent even if app server clock is wrong
- Survives application restarts

#### Decision 3: Indexes for common queries

```python
__table_args__ = (
    Index("idx_user_age", "age"),
    Index("idx_user_gender", "gender"),
    Index("idx_user_location", "location"),
    Index("idx_user_created_at", "created_at"),
)
```

**Why these indexes?**

Common query patterns:
```sql
-- Filtering candidates by age/gender/location
SELECT * FROM users
WHERE age BETWEEN 25 AND 35
  AND gender = 'female'
  AND location = 'San Francisco'

-- Getting recently active users
SELECT * FROM users
WHERE last_active_at > NOW() - INTERVAL '7 days'
```

**Index Strategy**:
- Index columns used in WHERE clauses
- Trade-off: Faster reads, slower writes (index must be updated)
- For dating app: Reads >> Writes (users browse more than create profiles)

---

## Entity #2: Interaction (The Training Data Gold Mine)

### Schema Design

```python
class Interaction(SQLModel, table=True):
    id: Optional[int]
    user_id: int                        # Who interacted
    target_user_id: int                 # With whom
    interaction_type: InteractionType   # like/dislike/super_like/block
    context: Optional[dict]             # JSON metadata
    timestamp: datetime                 # When (critical for temporal features)
```

### Design Decisions

#### Decision 1: Self-referential many-to-many through interaction table

**What is self-referential?**
- User interacts with User
- Same table on both sides of relationship

```
User A ──likes──> User B
User B ──likes──> User A  (mutual match!)
User C ──dislikes──> User A
```

**Why not separate tables for likes/dislikes?**
```sql
-- Bad: Separate tables
likes (user_id, target_id, timestamp)
dislikes (user_id, target_id, timestamp)
super_likes (user_id, target_id, timestamp)

-- Good: Single table with type
interactions (user_id, target_id, type, timestamp)
```

**Benefits**:
- ✅ Single query for all interactions
- ✅ Easier to analyze interaction patterns
- ✅ Can add new interaction types without schema changes
- ✅ ML training: All training data in one table

#### Decision 2: Enum for interaction_type

```python
class InteractionType(str, Enum):
    LIKE = "like"
    DISLIKE = "dislike"
    SUPER_LIKE = "super_like"
    BLOCK = "block"
```

**Why Enum instead of plain string?**
- ✅ **Type safety**: Can't accidentally insert "lik" or "Like" (typos)
- ✅ **Database constraint**: PostgreSQL enforces valid values
- ✅ **Self-documenting**: IDE autocomplete shows all options

**Database representation**:
```sql
ALTER TABLE interactions
ADD CONSTRAINT interaction_type_check
CHECK (interaction_type IN ('like', 'dislike', 'super_like', 'block'));
```

#### Decision 3: Unique constraint prevents duplicate interactions

```python
__table_args__ = (
    Index("idx_unique_interaction", "user_id", "target_user_id", unique=True),
)
```

**What this prevents**:
```sql
-- User 1 likes User 2 multiple times (not allowed)
INSERT INTO interactions (user_id, target_id, type) VALUES (1, 2, 'like');
INSERT INTO interactions (user_id, target_id, type) VALUES (1, 2, 'like');  -- ERROR!
```

**Why prevent duplicates?**
- ✅ Data integrity: One interaction per user pair
- ✅ ML training: Prevents duplicate training samples
- ✅ Business logic: User can't spam likes

**What if user changes their mind?**
```sql
-- Update existing interaction
UPDATE interactions
SET interaction_type = 'dislike', timestamp = NOW()
WHERE user_id = 1 AND target_user_id = 2;
```

#### Decision 4: Context field for ML metadata

```python
context: Optional[dict] = Field(default=None, sa_column=Column(JSON))
```

**What goes in context?**
```json
{
  "source": "recommendation",  // Where did they see this profile?
  "position": 3,              // Position in recommendation list
  "session_id": "abc123",     // User session tracking
  "device": "mobile"          // Device type
}
```

**Why store this?**
- ✅ **ML features**: Position bias (users like profiles shown first)
- ✅ **Analytics**: Which recommendation strategy works best?
- ✅ **A/B testing**: Track algorithm performance
- ✅ **Debugging**: Reproduce issues

---

## Entity #3: UserEmbedding (Feature Store Pattern)

### Schema Design

```python
class UserEmbedding(SQLModel, table=True):
    id: Optional[int]
    user_id: int                        # Foreign key to users (unique!)
    embedding_vector: List[float]       # JSON array of 64 floats
    model_version: str                  # "v1.0.0"
    created_at: datetime
```

### What is an Embedding?

**Embedding**: Dense vector representation of a user for similarity search

```python
# User 1: Likes hiking, photography, travel
embedding_user_1 = [0.8, 0.3, 0.1, 0.9, ..., 0.2]  # 64 dimensions

# User 2: Also likes hiking, photography
embedding_user_2 = [0.7, 0.4, 0.2, 0.8, ..., 0.3]  # Similar!

# Cosine similarity: 0.92 (very similar users)
```

**How it's used**:
```python
# Find similar users (collaborative filtering)
similar_users = faiss_index.search(embedding_user_1, k=100)
# Returns 100 most similar users in milliseconds!
```

### Design Decisions

#### Decision 1: Separate table for embeddings

**Why not store in users table?**

```python
# Bad: Embeddings in users table
class User(SQLModel, table=True):
    id: int
    name: str
    embedding_vector: List[float]  # 64 floats = 256 bytes minimum
```

**Problems**:
- ❌ Users table bloated (256+ bytes per row)
- ❌ Every user query fetches embeddings (slow)
- ❌ Can't have multiple embedding versions

**Good: Separate table**
```python
class UserEmbedding(SQLModel, table=True):
    user_id: int                        # 1:1 with User
    embedding_vector: List[float]       # Only fetched when needed
```

**Benefits**:
- ✅ Users table stays small and fast
- ✅ Only fetch embeddings when doing collaborative filtering
- ✅ Can store multiple versions for A/B testing

#### Decision 2: Model versioning

```python
model_version: str = Field(max_length=50)
```

**Why version embeddings?**

**Scenario**: You train a new embedding model
- Old model: "v1.0" (currently in production)
- New model: "v2.0" (testing in staging)

**Without versioning**:
```sql
-- All users get new embeddings immediately
UPDATE user_embeddings SET embedding_vector = new_vector;
-- If new model is worse, can't roll back!
```

**With versioning**:
```sql
-- Keep both versions
INSERT INTO user_embeddings (user_id, embedding_vector, model_version)
VALUES (1, new_vector, 'v2.0');

-- Old version still exists
SELECT * FROM user_embeddings WHERE model_version = 'v1.0';
```

**Benefits**:
- ✅ **A/B testing**: 50% users get v1.0, 50% get v2.0
- ✅ **Rollback**: If v2.0 is worse, switch back to v1.0
- ✅ **Gradual rollout**: Slowly migrate users to new version
- ✅ **Debugging**: Compare old vs new embeddings

#### Decision 3: JSON vs pgvector

```python
# Current: JSON array
embedding_vector: List[float] = Field(sa_column=Column(JSON))

# TODO comment in code:
# TODO: Replace JSON with pgvector type for efficient similarity search
```

**Why JSON now?**
- ✅ Simple: Works out of the box, no PostgreSQL extension
- ✅ Flexible: Easy to change embedding dimensions
- ❌ Slow similarity search: Must load all embeddings into Python/FAISS

**Why pgvector later?**
```sql
-- With pgvector extension
CREATE EXTENSION vector;

CREATE TABLE user_embeddings (
    user_id int,
    embedding_vector vector(64)  -- Native vector type!
);

-- Similarity search IN DATABASE (fast!)
SELECT user_id, embedding_vector <=> query_vector AS distance
FROM user_embeddings
ORDER BY distance
LIMIT 100;
```

**Benefits of pgvector**:
- ✅ Similarity search in database (no FAISS needed)
- ✅ Index support (HNSW, IVFFlat) for fast search
- ✅ Reduced memory (don't load all embeddings into app)

**Why not use it yet?**
- ❌ Requires PostgreSQL extension (deployment complexity)
- ❌ FAISS works fine for now (loaded once at startup)
- ❌ pgvector is newer, less battle-tested

**Migration path**: JSON → pgvector when scale demands it

---

## Entity #4: UserFeatures (Feature Store for ML)

### Schema Design

```python
class UserFeatures(SQLModel, table=True):
    id: Optional[int]
    user_id: int
    feature_set: dict                   # JSON: all computed features
    computed_at: datetime               # When features were computed
    version: str                        # Feature schema version
```

### What is a Feature Store?

**Problem**: Training-serving skew

```python
# Training (offline)
def compute_features_for_training(user):
    return {
        "age_normalized": user.age / 100,
        "account_age_days": (now() - user.created_at).days,
        "like_rate": user.likes / user.total_interactions
    }

# Serving (online, 6 months later)
def compute_features_for_serving(user):
    return {
        "age_normalized": user.age / 100,
        "account_age_days": (now() - user.created).days,  # Bug! created vs created_at
        "like_rate": user.likes / user.interactions       # Bug! total_interactions missing
    }
    # Features don't match! Model performs poorly.
```

**Solution**: Feature Store (single source of truth)

```python
class FeatureService:
    def compute_features(self, user):
        """Used for BOTH training AND serving"""
        return {
            "age_normalized": user.age / 100,
            "account_age_days": (datetime.now() - user.created_at).days,
            "like_rate": user.get_like_rate()
        }
```

**Store features**:
```sql
INSERT INTO user_features (user_id, feature_set, version)
VALUES (1, '{"age_normalized": 0.25, ...}', 'v1.0');
```

### Design Decisions

#### Decision 1: JSON for flexible feature schema

```python
feature_set: dict = Field(sa_column=Column(JSON))
```

**What goes in feature_set?**
```json
{
  "demographic": {
    "age_normalized": 0.25,
    "gender_encoded": 1,
    "location_population": 800000
  },
  "behavioral": {
    "like_rate": 0.15,
    "avg_session_duration_mins": 12.5,
    "days_since_last_active": 2
  },
  "collaborative": {
    "embedding_cluster": 3,
    "avg_similarity_to_liked_users": 0.72
  }
}
```

**Why JSON instead of columns?**

**Alternative**: One column per feature
```sql
CREATE TABLE user_features (
    user_id int,
    age_normalized float,
    gender_encoded int,
    like_rate float,
    avg_session_duration float,
    -- 50+ columns...
);
```

**Problems with columns**:
- ❌ Schema change for every new feature
- ❌ 50+ columns = wide table (hard to manage)
- ❌ Sparse data (many NULLs if feature doesn't apply)

**Benefits of JSON**:
- ✅ Add features without schema migration
- ✅ Feature versioning easy
- ✅ Group related features
- ✅ Flexible: Different features per user type

**Trade-off**: Can't index individual features easily
- For analytics queries, this would be bad
- For ML serving (fetch all features for one user), this is fine

#### Decision 2: Feature versioning

```python
version: str = Field(max_length=50)
```

**Why version features?**

**Scenario**: You add a new feature
```json
// v1.0 features
{
  "age_normalized": 0.25,
  "like_rate": 0.15
}

// v2.0 features (added interaction_diversity)
{
  "age_normalized": 0.25,
  "like_rate": 0.15,
  "interaction_diversity": 0.82  // New!
}
```

**Without versioning**:
- Model trained on v1.0 expects 2 features
- Production serves v2.0 with 3 features
- Model fails or performs poorly

**With versioning**:
```sql
-- Fetch features matching model version
SELECT feature_set FROM user_features
WHERE user_id = 1 AND version = 'v1.0';
```

**Benefits**:
- ✅ Training-serving consistency
- ✅ Can test new features before deploying
- ✅ Rollback if new features hurt performance

#### Decision 3: Computed_at timestamp

```python
computed_at: datetime
```

**Why track when features were computed?**

**Cache invalidation**:
```python
# Fetch features if computed recently
features = get_cached_features(user_id)
if features.computed_at < datetime.now() - timedelta(hours=1):
    # Re-compute stale features
    features = compute_fresh_features(user_id)
```

**Benefits**:
- ✅ Know if features are stale
- ✅ TTL-based caching
- ✅ Debug: Why did model perform badly? (Check if features were outdated)

---

## Entity #5: MLModel (Model Versioning & Persistence)

### Schema Design

```python
class MLModel(SQLModel, table=True):
    id: Optional[int]
    model_type: str                     # "ranking", "embedding"
    version: str                        # "v1.0.0"
    model_binary: bytes                 # Serialized model
    metrics: dict                       # {"accuracy": 0.85, "auc": 0.92}
    hyperparameters: dict               # {"learning_rate": 0.01}
    created_at: datetime
    is_active: bool                     # Which model is currently used?
```

### Why Store Models in Database?

**Alternative 1**: Store models in files
```
/app/models/
├── ranking_v1.0.pkl
├── ranking_v1.1.pkl
└── ranking_v2.0.pkl
```

**Alternative 2**: Store models in database (Chosen)

**Trade-offs**:
| Aspect | File System | Database |
|--------|-------------|----------|
| **Deployment** | Must deploy files with app | Just deploy app code |
| **Versioning** | Manual file management | Automatic with SQL |
| **Rollback** | Copy old file back | UPDATE is_active = TRUE |
| **Metadata** | Separate tracking needed | Stored with model |
| **Replication** | Must sync files across servers | Database replication handles it |

**Why database?**
- ✅ **Single source of truth**: Model + metadata together
- ✅ **Easy rollback**: Flip `is_active` flag
- ✅ **Multi-server deployment**: All servers fetch from same DB
- ✅ **A/B testing**: Serve different models to different users

### Design Decisions

#### Decision 1: Model versioning strategy

```python
model_type: str                         # What kind of model
version: str                            # Which version
is_active: bool                         # Currently serving?
```

**How versioning works**:

```sql
-- All ranking models
SELECT * FROM ml_models WHERE model_type = 'ranking';

-- Active ranking model
SELECT * FROM ml_models
WHERE model_type = 'ranking' AND is_active = TRUE;

-- Switch to new model (rollout)
UPDATE ml_models SET is_active = FALSE WHERE model_type = 'ranking';
UPDATE ml_models SET is_active = TRUE
WHERE model_type = 'ranking' AND version = 'v2.0';
```

**Benefits**:
- ✅ **A/B testing**: Set multiple models active, route users by ID
- ✅ **Rollback**: Set old version active again (instant)
- ✅ **Gradual rollout**: Slowly shift traffic to new model
- ✅ **History**: See all past models and their metrics

#### Decision 2: Store metrics with model

```python
metrics: dict = Field(sa_column=Column(JSON))
```

**What goes in metrics?**
```json
{
  "training": {
    "accuracy": 0.85,
    "auc_roc": 0.92,
    "precision": 0.78,
    "recall": 0.81,
    "loss": 0.23
  },
  "validation": {
    "accuracy": 0.83,
    "auc_roc": 0.90
  },
  "training_samples": 1000000,
  "training_duration_seconds": 3600
}
```

**Why store metrics?**
- ✅ **Model selection**: Compare models side-by-side
- ✅ **Debugging**: Why is v2.0 performing worse in production?
- ✅ **Reporting**: Track model improvement over time
- ✅ **Audit**: Prove model meets accuracy requirements

#### Decision 3: Store hyperparameters

```python
hyperparameters: dict = Field(sa_column=Column(JSON))
```

**What goes in hyperparameters?**
```json
{
  "learning_rate": 0.01,
  "num_trees": 100,
  "max_depth": 5,
  "regularization": 0.001,
  "feature_version": "v1.0"
}
```

**Why store hyperparameters?**
- ✅ **Reproducibility**: Re-train model with same settings
- ✅ **Debugging**: Check if hyperparameter change caused regression
- ✅ **Experiment tracking**: What settings worked best?
- ✅ **Compliance**: Audit trail for model decisions

---

## Storage Decision: Why PostgreSQL for Everything?

### The One-Database Strategy

**Current architecture**:
```
┌─────────────────────────────────────┐
│         PostgreSQL                  │
│                                     │
│  ├── users                          │
│  ├── interactions                   │
│  ├── user_embeddings                │
│  ├── user_features                  │
│  └── ml_models                      │
└─────────────────────────────────────┘
```

**Alternative**: Separate stores for different data types

```
┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐
│PostgreSQL │  │  Redis    │  │  S3/Blob  │  │  Vector   │
│           │  │           │  │  Storage  │  │    DB     │
│- users    │  │- features │  │- models   │  │- embeddings│
│- interact.│  │  (cache)  │  │  (files)  │  │  (search) │
└───────────┘  └───────────┘  └───────────┘  └───────────┘
```

### Why PostgreSQL for All (For Now)?

#### Reason 1: Operational Simplicity

**One database to manage**:
- ✅ One backup strategy
- ✅ One monitoring setup
- ✅ One connection pool
- ✅ One team to own it

**Multiple databases**:
- ❌ Backup each separately
- ❌ Monitor each separately
- ❌ Connection pools for each
- ❌ Consistency across stores is hard

#### Reason 2: ACID Transactions

**Example**: Create user + compute initial features
```python
async with session.begin():
    # Create user
    user = User(name="Alice", age=25)
    session.add(user)
    await session.flush()  # Get user.id

    # Compute initial features
    features = UserFeatures(user_id=user.id, feature_set={...})
    session.add(features)

    # Either both succeed or both rollback (atomic!)
```

**With separate stores**:
```python
# PostgreSQL
user = create_user_in_postgres()

# Redis
try:
    create_features_in_redis(user.id)
except:
    # User created but features failed!
    # Now we need manual cleanup or eventual consistency
    pass
```

#### Reason 3: PostgreSQL is Good Enough

**PostgreSQL capabilities**:
- ✅ **JSON support**: Store embeddings, features, model metadata
- ✅ **Full-text search**: For bio/interests matching
- ✅ **Async support**: Non-blocking with asyncpg
- ✅ **Scales vertically**: 100K users easily handled
- ✅ **Extensions available**: pgvector for similarity search

**When to add specialized stores**:
- **Redis**: When cache hit rate matters more than simplicity
- **Elasticsearch**: When complex text search is required
- **Vector DB**: When millions of embeddings need real-time search
- **S3**: When models are 100s of MBs

#### Reason 4: Premature Optimization Avoided

**Current needs** (from Phase 1):
- 100K-1M users
- 10K-100K requests per minute
- 300ms latency target

**PostgreSQL can handle**:
- ✅ Millions of rows easily
- ✅ 10K+ queries per second (with proper indexing)
- ✅ <10ms query latency (simple indexed queries)

**When we'd need specialized stores**:
- 10M+ users → Sharding or distributed database
- 100K+ QPS → Read replicas + Redis caching
- Real-time embedding search at scale → Vector database

### Trade-offs Table

| Storage Strategy | Pros | Cons | When to Use |
|-----------------|------|------|-------------|
| **PostgreSQL only** (current) | Simple, ACID, one backup | Limited by single DB | <1M users, <10K QPS |
| **PostgreSQL + Redis** | Fast cache, session storage | Cache invalidation complexity | >10K QPS, <100ms latency needed |
| **PostgreSQL + S3** | Unlimited model storage | S3 latency for loads | Models >10MB, many versions |
| **PostgreSQL + Vector DB** | Fast similarity search | Another system to manage | >1M users, real-time embeddings |

---

## Caching Strategy: In-Memory → Redis

### Current: In-Memory Caching

```python
# app/services/recommendation_service.py
class RecommendationService:
    def __init__(self):
        # Simple Python dictionary
        self._recommendation_cache: Dict[str, Tuple[List[int], datetime]] = {}
        self._cache_ttl = timedelta(minutes=10)
```

**How it works**:
```python
cache_key = f"recs:{user_id}"

# Check cache
if cache_key in self._recommendation_cache:
    recs, cached_at = self._recommendation_cache[cache_key]
    if datetime.now() - cached_at < self._cache_ttl:
        return recs  # Cache hit!

# Cache miss: generate recommendations
recs = await self.generate_recommendations(user_id)

# Store in cache
self._recommendation_cache[cache_key] = (recs, datetime.now())
```

### Why In-Memory First?

#### Benefits of In-Memory
- ✅ **Simple**: No external dependency
- ✅ **Fast**: Microsecond latency
- ✅ **No network**: Local memory access
- ✅ **No deployment complexity**: Works out of the box

#### Limitations of In-Memory
- ❌ **Per-process**: Each server has separate cache
- ❌ **Not persistent**: Restarts clear cache
- ❌ **Memory limited**: Can't cache millions of entries
- ❌ **Cold start**: New servers have empty cache

### Migration Path: In-Memory → Redis

**Why Redis eventually**:
```
In-Memory (Single Server)          Redis (Shared Cache)
┌───────────────────┐              ┌──────────────────┐
│  App Server 1     │              │  App Server 1    │
│  ┌─────────────┐  │              │                  │
│  │Local Cache  │  │◄──┐          └────────┬─────────┘
│  └─────────────┘  │   │                   │
└───────────────────┘   │                   ↓
                        │          ┌──────────────────┐
                        │          │      Redis       │
                        │          │  (Shared Cache)  │
                        │          └────────┬─────────┘
┌───────────────────┐   │                   │
│  App Server 2     │   │                   │
│  ┌─────────────┐  │   │          ┌────────▼─────────┐
│  │Local Cache  │  │◄──┘          │  App Server 2    │
│  └─────────────┘  │              └──────────────────┘
└───────────────────┘

Cache miss on Server 2              Cache hit on Server 2
even if Server 1 has it            because Redis is shared
```

**When to migrate to Redis**:
1. **Multiple servers**: Load balancer across 2+ app servers
2. **Cache efficiency matters**: Want to avoid redundant computation
3. **Persistent cache**: Keep cache warm across restarts
4. **Centralized invalidation**: Clear cache for specific users

**Redis configuration** (already in docker-compose):
```yaml
# docker-compose.yml
redis:
  image: redis:7-alpine
  ports:
    - "6379:6379"
```

**Migration is simple**:
```python
# Before: In-memory
self._cache = {}

# After: Redis
import aioredis
self._cache = await aioredis.from_url("redis://localhost:6379")

# API stays the same!
cached = await self._cache.get(cache_key)
```

---

## Database Connection Management

### Connection Pooling Configuration

```python
# app.core.db.py
engine = create_async_engine(
    str(settings.DATABASE_URL),
    pool_size=10,                       # Base connections
    max_overflow=20,                    # Extra connections under load
    pool_pre_ping=True,                 # Test connections before use
)
```

### What is Connection Pooling?

**Without pooling**:
```
Request 1 → Open connection → Query → Close connection
Request 2 → Open connection → Query → Close connection
Request 3 → Open connection → Query → Close connection
```
❌ Opening/closing connections is expensive (100ms each)

**With pooling**:
```
App Startup:
  Create 10 connections to database
  Keep them open in pool

Request 1 → Borrow connection from pool → Query → Return to pool
Request 2 → Borrow connection from pool → Query → Return to pool
Request 3 → Borrow connection from pool → Query → Return to pool
```
✅ Reuse connections (~1ms overhead)

### Configuration Decisions

#### pool_size=10
- 10 connections ready at all times
- For 10K QPS with 10ms queries: Need ~100 connections
- But we use async! One connection handles multiple requests

**With async**:
```
Connection 1:
  Request A: Query started → waiting for DB
  Request B: Query started → waiting for DB
  Request C: Query started → waiting for DB
  (10+ requests "in flight" per connection)
```

#### max_overflow=20
- Under high load, create up to 20 additional connections
- Total max: 10 (pool_size) + 20 (overflow) = 30 connections

**Why not just set pool_size=30?**
- Connections consume resources (memory, database slots)
- Most of the time, 10 is enough
- Only use extra connections during traffic spikes

#### pool_pre_ping=True
- Before using connection, ping database to check if alive
- Prevents errors from stale connections

**Without pre_ping**:
```
1. Get connection from pool
2. Try to query
3. Error: connection was closed (database restarted)
4. Retry logic needed
```

**With pre_ping**:
```
1. Get connection from pool
2. Ping: "Are you alive?" → No response
3. Discard dead connection, create new one
4. Query succeeds
```

---

## Interview Narratives

### Q: "Walk me through your data model design"

**Strong Answer (60 seconds)**:

> "We have 5 core entities: User, Interaction, UserEmbedding, UserFeatures, and MLModel.
>
> **User** stores profile data - name, age, location, interests. We use JSON for the interests array instead of normalization because it's faster for reads (no JOINs) and flexible for schema changes. This directly addresses our 300ms latency target.
>
> **Interaction** is the gold mine - it's our training data. It's a self-referential many-to-many table with a unique constraint to prevent duplicate interactions. We store interaction type as an enum for type safety, and context as JSON for ML metadata like position bias.
>
> **UserEmbedding and UserFeatures** implement the feature store pattern. They're separate tables with versioning, which ensures training-serving consistency. We can deploy new feature versions, A/B test them, and roll back if needed.
>
> **MLModel** stores serialized models with metrics and hyperparameters for versioning and rollback. We use an is_active flag to control which model serves traffic.
>
> Everything lives in PostgreSQL for operational simplicity - one database to backup, monitor, and manage."

---

### Q: "Why PostgreSQL for everything? Why not Redis for cache, S3 for models?"

**Strong Answer (45 seconds)**:

> "Operational simplicity at our scale. PostgreSQL handles our requirements: 1M users, 10K QPS, 300ms latency. It has JSON support for flexible schemas, full-text search, and async support.
>
> We could add Redis for caching, but right now in-memory caching works fine. We'd migrate to Redis when we have multiple servers and want a shared cache.
>
> For models, PostgreSQL is fine because our models are small (~1MB). If we had 100MB models or hundreds of versions, we'd move to S3. But for now, storing models in the database gives us easy versioning, atomic rollback, and automatic replication.
>
> The trade-off is: more specialized stores would be marginally faster, but we'd have 3-4 systems to manage instead of 1. We'll add them when scale demands it, not before."

---

### Q: "How do you ensure training-serving consistency?"

**Strong Answer (30 seconds)**:

> "Feature store pattern with versioning. UserFeatures table stores pre-computed features with a version field. The same FeatureService computes features for both training and serving.
>
> When we train a model, we record which feature version was used in the model metadata. At serving time, we fetch features matching that version. This prevents training-serving skew.
>
> We also version UserEmbeddings the same way - embeddings from model v1.0 don't mix with v2.0."

---

### Q: "Explain your indexing strategy"

**Strong Answer (30 seconds)**:

> "We index columns used in WHERE clauses for common queries. For Users: age, gender, location, created_at - because candidate generation filters on these. For Interactions: composite index on (user_id, timestamp) for temporal queries, and unique index on (user_id, target_user_id) to prevent duplicates.
>
> Trade-off is slower writes, but for a dating app, reads outnumber writes 100:1, so it's worth it. We measure query performance in development and add indexes when queries exceed 50ms."

---

## Summary: Data Decisions Map to Requirements

| Requirement (Phase 1) | Data Decision (Phase 3) | Why |
|----------------------|------------------------|-----|
| **Latency <300ms** | JSON arrays (no JOINs), Indexes | Single-table lookups are fast |
| **Training-serving consistency** | Feature store with versioning | Same features for training & serving |
| **A/B testing models** | MLModel versioning, is_active flag | Switch models instantly |
| **Scale (1M users)** | PostgreSQL + connection pooling | Handles scale, async for concurrency |
| **Operational simplicity** | Single database (PostgreSQL) | One system to manage |
| **Data integrity** | ACID transactions, unique constraints | Prevent duplicate interactions |

---

## Next Steps
- [ ] Validate understanding by exploring actual database schema
- [ ] Move to Phase 4: ML Pipeline Architecture (Two-Stage Design)
- [ ] Understand how feature store enables ML workflow
