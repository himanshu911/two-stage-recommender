# Phase 4: ML Pipeline Architecture (Two-Stage Design)

## Overview: The Core ML System

This is **the heart of the system** - where ML meets production engineering. We'll answer:
1. Why two-stage instead of single-stage?
2. How does Stage 1 (Candidate Generation) work?
3. How does Stage 2 (Ranking) work?
4. Why these specific algorithms (matrix factorization + FAISS + logistic regression)?
5. How do we ensure training-serving consistency?

---

## The Central Question: Why Two-Stage?

### The Scale Problem

**Naive Approach** (Single-Stage):
```python
# For each user, score ALL other users with ML model
def get_recommendations(user_id):
    all_users = get_all_users()  # 1 million users

    scores = []
    for candidate in all_users:
        # Expensive ML model (neural network, 50ms per prediction)
        score = ml_model.predict(user_id, candidate)
        scores.append((candidate, score))

    return sorted(scores, reverse=True)[:20]

# Time: 1M users × 50ms = 50,000 seconds = 14 hours! ❌
```

**Why this fails**:
- ❌ **Too slow**: 1M users ×  50ms = 50,000 seconds
- ❌ **Too expensive**: GPU/CPU cost for 1M predictions per request
- ❌ **Doesn't meet latency target**: Need <300ms, this takes hours

---

### The Two-Stage Solution

```
┌──────────────────────────────────────────────────────────────┐
│                  1 Million Users                              │
└────────────────────────┬─────────────────────────────────────┘
                         │
        ┌────────────────▼────────────────┐
        │   Stage 1: Candidate Generation  │  Fast, approximate
        │   - Collaborative Filtering      │  100ms
        │   - Content-Based                │
        │   - Exploration                  │
        └────────────────┬─────────────────┘
                         │
                    100 candidates
                         │
        ┌────────────────▼────────────────┐
        │   Stage 2: Ranking              │  Accurate, expensive
        │   - ML Model (Logistic Reg)     │  50ms
        │   - Feature Engineering          │
        │   - Probability Scoring          │
        └────────────────┬─────────────────┘
                         │
                    Top 20 recommendations
                         │
                         ▼
                      User

Total Time: 100ms + 50ms = 150ms ✓
```

**Why this works**:
- ✅ **Stage 1**: Fast approximate methods narrow down 1M → 100 (100ms)
- ✅ **Stage 2**: Expensive ML only scores 100 candidates (50ms)
- ✅ **Total**: 150ms << 300ms target ✓

---

## Stage 1: Candidate Generation (The Funnel)

### Objective
**Reduce 1M users to ~100 candidates quickly**

### Three Strategies (Diversity)

```python
# app/services/candidate_generation.py
async def generate_candidates(user_id: int, limit: int = 50):
    candidates = []

    # 1. Collaborative Filtering (50% of candidates)
    cf_candidates = await self._generate_cf_candidates(user_id, limit // 2)
    candidates.extend(cf_candidates)

    # 2. Content-Based (33% of candidates)
    content_candidates = await self._generate_content_candidates(user_id, limit // 3)
    candidates.extend(content_candidates)

    # 3. Random Exploration (remaining candidates)
    random_candidates = await self._generate_random_candidates(user_id, limit - len(candidates))
    candidates.extend(random_candidates)

    return deduplicate_and_rank(candidates)
```

**Why multiple strategies?**
- **Collaborative Filtering**: "Users like you also liked..."
- **Content-Based**: "Users similar to your profile..."
- **Random**: Diversity, serendipity, avoid filter bubbles

---

### Strategy 1: Collaborative Filtering (Matrix Factorization + FAISS)

#### What is Collaborative Filtering?

**Idea**: If User A and User B liked similar people, they have similar tastes

```
User-Item Interaction Matrix:

           Person1  Person2  Person3  Person4
User A        1        0        1        0     (liked 1, 3)
User B        1        0        1        1     (liked 1, 3, 4)
User C        0        1        0        0     (liked 2)

→ User A and User B are similar (both liked 1 and 3)
→ Recommend Person 4 to User A (because B liked it)
```

#### Matrix Factorization (Convert Users to Vectors)

**Goal**: Represent each user as a 64-dimensional vector (embedding)

```
User A = [0.8, 0.3, 0.1, 0.9, ..., 0.2]  (64 numbers)
User B = [0.7, 0.4, 0.2, 0.8, ..., 0.3]  (similar to A!)
User C = [-0.5, 0.9, -0.2, 0.1, ..., 0.7] (different!)
```

**How to learn these vectors?** Minimize reconstruction error

```python
# Training objective
for each interaction (user, item):
    prediction = dot(user_embedding, item_embedding)
    error = actual_interaction - prediction

    # Update embeddings to reduce error
    user_embedding += learning_rate * error * item_embedding
    item_embedding += learning_rate * error * user_embedding
```

**Code**:
```python
class CollaborativeFilteringModel:
    def train_offline(self, user_item_matrix, user_ids, item_ids, epochs=100):
        # Initialize random embeddings (64 dimensions)
        self.user_embeddings = {
            user_id: np.random.normal(0, 0.1, 64)
            for user_id in user_ids
        }

        # SGD training loop
        for epoch in range(epochs):
            for i, user_id in enumerate(user_ids):
                for j, item_id in enumerate(item_ids):
                    if user_item_matrix[i, j] > 0:  # Observed interaction
                        # Predict
                        user_emb = self.user_embeddings[user_id]
                        item_emb = self.item_embeddings[item_id]
                        prediction = np.dot(user_emb, item_emb)

                        # Calculate error
                        error = user_item_matrix[i, j] - prediction

                        # Update embeddings (SGD)
                        self.user_embeddings[user_id] += lr * (error * item_emb - reg * user_emb)
                        self.item_embeddings[item_id] += lr * (error * user_emb - reg * item_emb)
```

#### FAISS: Fast Similarity Search

**Problem**: Given User A's embedding, find 100 most similar users
- **Naive**: Compare to all 1M users → O(1M) comparisons → too slow
- **FAISS**: Approximate nearest neighbor search → O(log N) → fast!

**What is FAISS?**
- Facebook AI Similarity Search
- Optimized library for vector similarity search
- Uses indexing structures (like database indexes, but for vectors)

**Code**:
```python
def _build_index(self):
    # Convert embeddings to matrix
    embeddings_matrix = np.array([
        self.user_embeddings[user_id]
        for user_id in user_ids
    ]).astype(np.float32)

    # Build FAISS index (Inner Product = Cosine Similarity)
    self.user_index = faiss.IndexFlatIP(64)  # 64 dimensions
    self.user_index.add(embeddings_matrix)  # Add all user embeddings

def find_similar_users(self, user_id, k=100):
    # Get user embedding
    user_embedding = self.user_embeddings[user_id]

    # Fast similarity search (milliseconds!)
    similarities, indices = self.user_index.search(user_embedding.reshape(1, -1), k)

    # Returns top-k similar users in ~1ms
    return [(self.user_id_mapping[idx], similarity)
            for idx, similarity in zip(indices[0], similarities[0])]
```

**Why FAISS?**
- ✅ **Speed**: Finds 100 similar users among 1M in ~1-10ms
- ✅ **Scalability**: Scales to billions of vectors
- ✅ **GPU support**: Can use GPU for even faster search
- ✅ **Battle-tested**: Used by Facebook, Uber, Spotify

**Trade-offs**:
| Aspect | Exact Search | FAISS (Approximate) |
|--------|--------------|---------------------|
| **Accuracy** | 100% accurate | ~95-99% accurate |
| **Speed** | O(N) - slow | O(log N) - fast |
| **Memory** | Loads all embeddings | Compressed indexes |
| **When to use** | <10K users | >10K users |

---

### Strategy 2: Content-Based Filtering

**Idea**: Match based on profile attributes (age, location, interests)

```python
async def _generate_content_candidates(self, user_id, limit, exclude_users, filters):
    # Get target user
    user = await self.user_repository.get_by_id(user_id)

    # Query similar users by attributes
    query = select(User).where(
        User.id.not_in(exclude_users),
        User.age.between(user.age - 5, user.age + 5),  # Similar age
        User.location == user.location  # Same location
    )

    potential_candidates = await self.session.execute(query)

    # Score by content similarity
    for candidate in potential_candidates:
        score = self._calculate_content_score(user, candidate)
        # score = age_similarity * 0.3 + location_match * 0.3 + interest_overlap * 0.4
```

**Content Score Calculation**:
```python
def _calculate_content_score(self, user_features, candidate):
    score = 0.0

    # 1. Age similarity (30% weight)
    age_diff = abs(user_features["age"] - candidate.age)
    age_score = max(0, 1 - (age_diff / 10))  # Decay over 10 years
    score += age_score * 0.3

    # 2. Location match (30% weight)
    if user_features["location"] == candidate.location:
        score += 0.3

    # 3. Interest overlap - Jaccard similarity (40% weight)
    user_interests = set(user_features["interests"])
    candidate_interests = set(candidate.interests)

    intersection = len(user_interests & candidate_interests)
    union = len(user_interests | candidate_interests)
    interest_score = intersection / union if union > 0 else 0
    score += interest_score * 0.4

    return score
```

**Why Content-Based?**
- ✅ **Cold start**: Works for new users with no interactions
- ✅ **Explainable**: "You both like hiking and photography"
- ✅ **Fast**: Simple SQL queries with indexes
- ❌ **Limited**: Doesn't capture latent preferences

---

### Strategy 3: Random Exploration

**Why random?**
- **Avoid filter bubbles**: User only sees people like them
- **Serendipity**: Unexpected good matches
- **Diversity**: Show variety of profiles

```python
async def _generate_random_candidates(self, user_id, limit, exclude_users, filters):
    query = (
        select(User)
        .where(User.id.not_in(exclude_users))
        .order_by(func.random())  # Database random ordering
        .limit(limit)
    )

    random_users = await self.session.execute(query)

    # Assign low base score (so they rank below CF and content)
    return [Candidate(user_id=user.id, score=0.1, source="random")
            for user in random_users]
```

**Exploration-Exploitation Trade-off**:
```
100 candidates breakdown:
- 50% Collaborative Filtering (Exploitation - show what they like)
- 33% Content-Based (Exploitation - similar profiles)
- 17% Random (Exploration - discover new patterns)
```

---

## Stage 2: Ranking (ML-Powered Precision)

### Objective
**Accurately score 100 candidates → return top 20**

### Why Logistic Regression?

**Alternatives considered**:
| Model | Pros | Cons | Why Not (Yet)? |
|-------|------|------|----------------|
| **Logistic Regression** (Chosen) | Fast inference, interpretable, stable | Limited capacity | ✓ **Good starting point** |
| **Gradient Boosted Trees** (XGBoost) | High accuracy, handles non-linear | Slower inference, harder to deploy | Next iteration |
| **Neural Network** (Deep Learning) | Highest capacity | Slow inference, needs GPU, hard to interpret | Future, at scale |
| **Heuristics** (Rule-based) | Fast, explainable | Not personalized, manual tuning | Fallback only |

**Why Logistic Regression wins (for now)**:

1. **Fast Inference** (Critical for 300ms budget)
```python
# Logistic regression: <1ms for 100 predictions
probabilities = model.predict_proba(feature_matrix)  # 100 × 9 features

# Neural network: ~10ms for 100 predictions
probabilities = neural_net.forward(feature_matrix)  # Too slow!
```

2. **Interpretable** (Understand what drives recommendations)
```python
# Feature importance from model coefficients
feature_importance = {
    "age": 0.45,              # Age is important!
    "like_rate": 0.32,        # User's like rate matters
    "account_age_days": 0.15, # Newer accounts ranked higher
    "interests_count": 0.08   # Profile completeness
}
```

3. **Stable in Production** (Doesn't require GPUs, easy to debug)

4. **Good Enough** (For initial launch, can upgrade later)

---

### Feature Engineering for Ranking

**Feature Schema** (9 features):
```python
self.feature_schema = [
    "age",                     # Candidate age (normalized)
    "account_age_days",        # How long candidate has been on platform
    "interests_count",         # Number of interests (profile completeness)
    "total_interactions",      # Candidate's total activity level
    "like_rate",               # Candidate's like rate (selectiveness)
    "recent_activity_30d",     # Candidate active recently?
    "activity_streak_days",    # Consecutive days of activity
    "embedding_norm",          # Norm of embedding vector
    "embedding_sparsity"       # Sparsity of embedding
]
```

**Why these features?**
- **Demographic**: age (compatibility)
- **Engagement**: recent_activity, activity_streak (active users get ranked higher)
- **Quality**: like_rate, interests_count (complete profiles rank higher)
- **Collaborative**: embedding_norm, embedding_sparsity (ML signals)

---

### Ranking Process

```python
async def rank_candidates(self, user_id, candidate_ids, limit=20):
    # 1. Extract features for each candidate
    candidate_features = []
    for candidate_id in candidate_ids:
        features = await self._get_candidate_features(user_id, candidate_id)
        feature_vector = self._features_to_vector(features)  # Convert to [9 numbers]
        candidate_features.append(feature_vector)

    # 2. Predict probabilities (vectorized, fast!)
    feature_matrix = np.array(candidate_features)  # Shape: (100, 9)
    probabilities = self.model.predict_proba(feature_matrix)  # Shape: (100,)

    # 3. Create ranked candidates with scores
    ranked_candidates = [
        RankedCandidate(
            user_id=candidate_id,
            score=probability,  # Probability of positive interaction
            features=feature_dict,
            model_version="v1"
        )
        for candidate_id, probability in zip(candidate_ids, probabilities)
    ]

    # 4. Sort by score and return top-K
    ranked_candidates.sort(key=lambda x: x.score, reverse=True)
    return ranked_candidates[:limit]
```

**What does the score mean?**
- `score = 0.85` → 85% probability user will like this candidate
- `score = 0.20` → 20% probability (low, rank lower)

---

### Training the Ranking Model

**Offline Training Process**:

```python
# 1. Collect training data from interactions
training_data = []
for interaction in past_interactions:
    # Positive sample (like/super_like)
    if interaction.type in ["like", "super_like"]:
        features = get_features(interaction.user_id, interaction.target_id)
        training_data.append((features, label=1))  # Positive

    # Negative sample (dislike/pass)
    elif interaction.type == "dislike":
        features = get_features(interaction.user_id, interaction.target_id)
        training_data.append((features, label=0))  # Negative

# 2. Train model
X = np.array([features for features, _ in training_data])
y = np.array([label for _, label in training_data])

model = LogisticRegression()
model.fit(X, y)

# 3. Evaluate
accuracy = model.score(X_test, y_test)  # e.g., 0.75

# 4. Save to database
model_binary = pickle.dumps(model)
MLModel.create(model_type="ranking", version="v1", model_binary=model_binary)
```

**Key Design Decisions**:

#### Decision 1: Feature Scaling
```python
class LogisticRegressionModel:
    def __init__(self):
        self.scaler = StandardScaler()  # Scale features to mean=0, std=1
        self.model = LogisticRegression()

    def train(self, X, y):
        X_scaled = self.scaler.fit_transform(X)  # Scale!
        self.model.fit(X_scaled, y)
```

**Why scale?**
- Age ranges from 18-80 (scale: ~60)
- like_rate ranges from 0-1 (scale: 1)
- Without scaling, age dominates the model
- StandardScaler makes all features equal scale

#### Decision 2: Regularization
```python
model = LogisticRegression(C=1.0)  # C = 1/lambda (regularization strength)
```

**Why regularize?**
- Prevents overfitting on training data
- C=1.0 is moderate regularization
- Smaller C → more regularization → simpler model

#### Decision 3: Model Versioning
```python
class RankingService:
    def __init__(self, model_version="v1"):
        self.model_version = model_version  # Can A/B test v1 vs v2

    async def _load_model(self):
        # Load from database
        model_data = await self.session.execute(
            select(MLModel)
            .where(
                MLModel.model_type == "ranking",
                MLModel.version == self.model_version,
                MLModel.is_active == True
            )
        )
```

**Why version models?**
- A/B test: 50% get v1, 50% get v2
- Rollback: If v2 is worse, switch back to v1
- Gradual rollout: Slowly migrate users to new model

---

## Feature Service: The Bridge (Training-Serving Consistency)

### The Problem: Training-Serving Skew

**Training (Offline, 6 months ago)**:
```python
def compute_features_for_training(user):
    return {
        "age": user.age,
        "account_age_days": (today - user.created_at).days,
        "like_rate": user.likes_count / user.total_interactions
    }

# Train model
features_train = [compute_features_for_training(u) for u in users]
model.fit(features_train, labels)
```

**Serving (Online, today)** - Developer makes a mistake:
```python
def compute_features_for_serving(user):
    return {
        "age": user.age / 100,  # BUG: Normalized! Training didn't normalize
        "account_age": (today - user.created).days,  # BUG: Different field name
        "like_rate": user.likes / user.swipes  # BUG: Different denominator
    }

# Predictions are garbage because features don't match!
```

---

### Solution: Feature Store Pattern

**Single Source of Truth**:
```python
# app/services/feature_service.py

class FeatureService:
    """Used for BOTH training AND serving"""

    async def get_features(self, user_id: int) -> Dict[str, Any]:
        # Check cache first
        cached = await self._get_cached_features(user_id)
        if cached and self._is_fresh(cached):
            return cached.feature_set

        # Extract features (same code for training and serving!)
        features = {}

        # Demographic features
        demo_extractor = DemographicFeatureExtractor(self.user_repository)
        features.update(await demo_extractor.extract(user_id))

        # Behavioral features
        behavior_extractor = BehavioralFeatureExtractor(self.interaction_repository)
        features.update(await behavior_extractor.extract(user_id))

        # Collaborative features
        collab_extractor = CollaborativeFeatureExtractor(self.session)
        features.update(await collab_extractor.extract(user_id))

        # Store in feature store (UserFeatures table)
        await self._store_features(user_id, features, version="v1.0")

        return features
```

**Feature Extractors** (Modular, Testable):
```python
class DemographicFeatureExtractor:
    async def extract(self, user_id: int) -> Dict[str, Any]:
        user = await self.user_repository.get_by_id(user_id)
        return {
            "age": user.age,
            "account_age_days": (datetime.utcnow() - user.created_at).days,
            "interests_count": len(user.interests)
        }

class BehavioralFeatureExtractor:
    async def extract(self, user_id: int) -> Dict[str, Any]:
        stats = await self.interaction_repository.get_stats(user_id)
        return {
            "total_interactions": stats["total"],
            "like_rate": stats["likes"] / max(stats["total"], 1),
            "recent_activity_30d": stats["recent_30d"]
        }
```

**Benefits**:
1. **Training-Serving Consistency**: Same code path for both
2. **Versioning**: Store features with version number
3. **Caching**: Compute once, use many times
4. **Testability**: Each extractor is independent
5. **Extensibility**: Easy to add new feature extractors

---

## Complete Pipeline Flow (End-to-End)

```
User Request: "Show me recommendations"
    │
    ▼
┌─────────────────────────────────────────────┐
│  RecommendationService.get_recommendations() │
└────────────────┬────────────────────────────┘
                 │
    ┌────────────▼────────────┐
    │  Check Cache (10min TTL) │
    └────────────┬────────────┘
                 │ Cache Miss
                 ▼
┌──────────────────────────────────────────────┐
│  Stage 1: CandidateGenerationService         │
│  ├─ Collaborative Filtering (50 candidates)  │ 100ms
│  ├─ Content-Based (33 candidates)            │
│  └─ Random Exploration (17 candidates)       │
│  → 100 candidates                            │
└────────────────┬─────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────┐
│  Stage 2: RankingService                     │
│  ├─ FeatureService.get_features()            │ 50ms
│  │  ├─ DemographicFeatureExtractor           │
│  │  ├─ BehavioralFeatureExtractor            │
│  │  └─ CollaborativeFeatureExtractor         │
│  ├─ Model.predict_proba(features)            │
│  └─ Sort by score                            │
│  → Top 20 ranked candidates                  │
└────────────────┬─────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────┐
│  RecommendationService                       │
│  ├─ Build response with metadata             │
│  ├─ Update cache                             │
│  └─ Log metrics                              │
└────────────────┬─────────────────────────────┘
                 │
                 ▼
        Return to User

Total Time: ~150ms (within 300ms budget ✓)
```

---

## Key Design Decisions & Trade-offs

### Decision 1: Why 100 candidates from Stage 1?

```
Too Few (10 candidates):
  ✓ Very fast
  ❌ Stage 2 has no variety to rank
  ❌ Poor recommendation quality

Just Right (100 candidates):
  ✓ Fast enough (~100ms)
  ✓ Enough variety for Stage 2
  ✓ Good quality recommendations

Too Many (1000 candidates):
  ❌ Slower Stage 1 (~500ms)
  ❌ Stage 2 takes longer (1000 predictions)
  ❌ Exceeds latency budget
```

**Tunable**: Can adjust based on performance monitoring

---

### Decision 2: Why Matrix Factorization (not deep learning)?

| Aspect | Matrix Factorization | Neural Collaborative Filtering |
|--------|---------------------|-------------------------------|
| **Training Time** | Minutes | Hours |
| **Inference Time** | 1ms | 10ms |
| **Interpretability** | Embedding vectors | Black box |
| **Data Requirements** | 1K interactions | 100K+ interactions |
| **GPU Needed** | No | Yes (for training) |
| **Our Choice** | ✓ **Chosen** | Future (when >1M users) |

**When to upgrade**: When we have 1M+ users and 100K+ daily interactions

---

### Decision 3: Why Logistic Regression (not XGBoost)?

| Aspect | Logistic Regression | XGBoost |
|--------|---------------------|---------|
| **Inference Speed** | <1ms for 100 predictions | ~5-10ms |
| **Model Size** | <1MB | ~10-100MB |
| **Interpretability** | High (coefficients) | Medium (feature importance) |
| **Accuracy** | Good | Better |
| **Deployment** | Simple | Needs XGBoost library |
| **Our Choice** | ✓ **Chosen for v1** | Next iteration |

**Migration path**: LR → XGBoost → Neural Network (as scale grows)

---

### Decision 4: Feature Store vs On-the-Fly Computation

**Alternative**: Compute features on every request (no caching)
```python
# Without feature store
async def rank_candidates(user_id, candidate_ids):
    for candidate_id in candidate_ids:
        features = compute_features(candidate_id)  # Recompute every time!
        # 100 candidates × 50ms = 5000ms ❌
```

**With Feature Store**:
```python
# With feature store
async def rank_candidates(user_id, candidate_ids):
    for candidate_id in candidate_ids:
        features = await feature_service.get_features(candidate_id)
        # Cached! 100 candidates × 1ms = 100ms ✓
```

**Trade-offs**:
| Aspect | On-the-Fly | Feature Store (Cached) |
|--------|-----------|----------------------|
| **Speed** | Slow (recompute) | Fast (cached) |
| **Storage** | No extra storage | UserFeatures table |
| **Freshness** | Always fresh | TTL-based (5min) |
| **Consistency** | Harder | Built-in versioning |

**Our choice**: Feature store with 5-minute TTL
- Fast enough for recommendations
- Fresh enough for user experience
- Consistent across training/serving

---

## Interview Narratives

### Q: "Walk me through your ML pipeline"

**Strong Answer (90 seconds)**:

> "We use a two-stage recommendation pipeline to balance latency and accuracy.
>
> **Stage 1 - Candidate Generation** (100ms): We start with 1 million users and quickly narrow down to 100 candidates using three strategies. First, collaborative filtering with matrix factorization - we train 64-dimensional user embeddings and use FAISS for fast similarity search. This gives us 50 candidates of 'users like you also liked.' Second, content-based filtering on profile attributes like age, location, and interests - this handles cold start and gives us 33 candidates. Third, random exploration for 17 candidates to avoid filter bubbles and enable serendipity.
>
> **Why multiple strategies?** Diversity - we want exploitation (show what they like) and exploration (discover new patterns). Also, collaborative filtering doesn't work for new users, so content-based fills that gap.
>
> **Stage 2 - Ranking** (50ms): We take those 100 candidates and score them with a logistic regression model. We extract 9 features per candidate - demographic (age), engagement (recent activity, streak), and quality signals (like rate, profile completeness). The model predicts the probability of a positive interaction, and we return the top 20.
>
> **Why two stages?** Can't afford to run ML model on 1 million users - that would take hours. By funneling down to 100 first, we keep total latency at 150ms, well within our 300ms budget."

---

### Q: "Why not use a neural network for ranking?"

**Strong Answer (45 seconds)**:

> "Inference speed and operational simplicity. Logistic regression gives us <1ms inference for 100 predictions. A neural network would be 5-10ms and require careful deployment (model serving, GPU management, version control).
>
> For our current scale - ~100K users, 10K daily interactions - logistic regression gives us good enough accuracy (75-80%) with much simpler operations. We can iterate faster, debug easier, and the model is interpretable (we can see feature coefficients).
>
> **When we'd upgrade**: When we hit 1M+ users with 100K+ daily interactions, and we have evidence that a neural network significantly improves metrics. At that point, the operational complexity is worth it. But not before."

---

### Q: "How do you ensure training-serving consistency?"

**Strong Answer (45 seconds)**:

> "Feature store pattern with a single FeatureService used for both training and serving. When we train offline, we call `FeatureService.get_features()` to extract features from historical data. When we serve online, we call the same function. Same code path means features match.
>
> We also version features - when we store them in the UserFeatures table, we record the version. The model metadata includes which feature version it was trained on. At serving time, we fetch features matching that version.
>
> The feature service has three extractors - demographic, behavioral, and collaborative - each responsible for one type of feature. This modular design makes it easy to test each extractor independently and ensure consistency."

---

### Q: "Why FAISS instead of database queries for similarity search?"

**Strong Answer (30 seconds)**:

> "Speed. To find 100 similar users among 1 million, a database query with vector similarity would be O(N) - we'd have to compare against all users. That's 1 million dot products, taking hundreds of milliseconds.
>
> FAISS builds an index structure (like a database B-tree, but for vectors) that makes similarity search O(log N). We can find 100 similar users in 1-10ms. For 1M users, that's a 100x speedup.
>
> Trade-off is approximate - FAISS might miss the absolute most similar user, but it gets 95-99% accuracy, which is fine for recommendations. We prioritize speed over perfect accuracy."

---

### Q: "What happens if the ML model isn't trained yet?"

**Strong Answer (30 seconds)**:

> "We have a fallback ranking strategy using simple heuristics. We score candidates based on profile completeness (has bio, has interests), recent activity (active in last 30 days), and account age (slight boost for newer users).
>
> This ensures the system works from day one, even before we have enough interaction data to train a good model. Once we have 100+ interactions, we train the collaborative filtering model. Once we have 1000+ interactions, we train the ranking model.
>
> Graceful degradation - the system works with reduced quality, rather than failing completely."

---

### Q: "How would you scale this to 10M users?"

**Strong Answer (60 seconds)**:

> "Three main bottlenecks to address:
>
> **1. FAISS index size**: 10M users × 64 dimensions × 4 bytes = 2.5GB. Still fits in memory, but we'd want GPU acceleration for search. FAISS supports GPU indexes for 10-100x speedup.
>
> **2. Feature computation**: With 10M users, we can't compute features on-the-fly. We'd need batch feature computation overnight, stored in the feature store with longer TTLs (1 hour vs 5 minutes). Use Spark or similar for distributed feature engineering.
>
> **3. Model complexity**: At 10M users with rich interaction data, we'd upgrade from logistic regression to XGBoost or a neural network. We'd also consider candidate generation sharding - split users into clusters and only search within cluster.
>
> **Infrastructure**: Add read replicas for the database, Redis cluster for caching, and potentially separate the ML inference into its own service with horizontal scaling."

---

## Summary: Why This Design?

| Requirement (Phase 1) | ML Design Decision (Phase 4) | Why |
|----------------------|------------------------------|-----|
| **Latency <300ms** | Two-stage pipeline (150ms total) | Can't ML-score 1M users in time |
| **Accuracy** | Stage 2 with ML model | Focus expensive ML on top candidates |
| **Scale (1M users)** | FAISS for similarity search | O(log N) vs O(N) |
| **Cold start** | Content-based + random exploration | Works without interaction history |
| **Training-serving consistency** | Feature store pattern | Single code path for both |
| **Operational simplicity** | Logistic regression (not neural net) | <1ms inference, no GPU needed |
| **Diversity** | 3 strategies (CF, content, random) | Avoid filter bubbles |
| **Interpretability** | Logistic regression coefficients | Debug why recommendations shown |
| **A/B testing** | Model versioning | Switch models, rollback if needed |

---

## Next Steps
- [ ] Validate understanding by exploring actual ML service code
- [ ] Move to Phase 5: Service Layer & API Design
- [ ] Connect everything: How API → Services → ML Pipeline → Data