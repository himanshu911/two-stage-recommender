# Phase 1: Problem Space Analysis
## Dating App Recommendation System

### The Core Problem
**Goal**: Help users discover potential romantic matches in a dating app (like Tinder, Bumble, Hinge).

**Challenge**: Among potentially millions of users, how do we show each user a personalized feed of 20-50 people they might be interested in meeting?

### What Makes This Problem Unique?

#### 1. Mutual Interest Required (Two-Sided Marketplace)
Unlike Netflix (one-sided: recommend movies TO user) or Amazon (one-sided: recommend products TO user):
- **Dating requires mutual match**: Both sides must be interested
- Can't just optimize for User A's preferences; User B must also find User A attractive
- This is a **fairness problem**: Popular users get over-recommended, others get ignored

#### 2. Real-Time Interaction Loop
- User swipes (like/dislike/super like)
- Immediate feedback signal
- Next recommendations should reflect this new information
- **Challenge**: How do we incorporate feedback fast enough?

#### 3. Exploration vs Exploitation
- **Exploitation**: Show users their "type" (similar to past likes)
- **Exploration**: Introduce diversity, serendipity, unexpected matches
- Too much exploitation → echo chamber, boredom
- Too much exploration → poor matches, user frustration

#### 4. Cold Start Problem (Severe)
- New users have NO interaction history
- Can't rely on collaborative filtering initially
- Must use content-based features: age, location, interests, bio
- **Challenge**: How do we blend collaborative + content-based?

### User Scenarios

#### Primary Flow
1. User opens app
2. System generates 20-50 personalized recommendations
3. User swipes through profiles (like/dislike/super like)
4. System tracks interactions
5. Next time: Better recommendations based on feedback

#### Secondary Flows
- View profile details
- Match notification (mutual like)
- Block/report users
- Update own profile → affects how others see you

### Functional Requirements

#### Core Features
1. **User Management**
   - Create profile (demographics, bio, interests, photos)
   - Update profile
   - Search/filter users (age range, location, etc.)

2. **Interaction Tracking**
   - Record likes, dislikes, super likes, blocks
   - Store context (where did interaction happen?)
   - Support undo/review past interactions

3. **Personalized Recommendations**
   - Generate 20-50 candidates per user per session
   - Consider user preferences (age, location, interests)
   - Incorporate past interaction history
   - Balance exploration and exploitation
   - Explain WHY we recommended someone (transparency)

4. **Matching**
   - Detect mutual likes
   - Notify both users

#### Filtering Requirements
- Respect user preferences (age range, distance, gender)
- Exclude already-interacted users (don't re-show people I already swiped)
- Exclude blocked users (in both directions)
- Respect privacy settings

### Non-Functional Requirements

#### 1. Latency (Critical for UX)
- **Target**: Generate recommendations in <300ms (p95)
- **Why**: User opens app → expect immediate feed
- **Challenge**: Can't afford to score millions of users in real-time
- **This is why we need efficient retrieval** → Two-stage design hint!

#### 2. Throughput
- **Scale**: Support 100K-1M concurrent users
- **QPM (Queries Per Minute)**: 10K-100K recommendation requests
- **Challenge**: How do we scale the ML inference?

#### 3. Accuracy / Relevance
- **Goal**: High match rate (if we recommend A to B, they should like each other)
- **Metrics**:
  - Like rate (% of recommendations that get liked)
  - Match rate (% of recommendations that become mutual matches)
  - User retention (do users come back?)

#### 4. Freshness
- **Requirement**: Incorporate new interactions within minutes
- **Why**: If I just liked 5 people with blue eyes, show me more brunettes with blue eyes NOW
- **Challenge**: Real-time feature updates vs batch model training

#### 5. Fairness & Diversity
- **Goal**: Don't just recommend the "hottest" users to everyone
- **Why**:
  - Overexposure for popular users
  - Poor experience for average users (never get matches)
  - Legal/ethical concerns (discrimination)
- **Challenge**: Balance relevance with fairness

#### 6. Privacy & Safety
- **Requirement**:
  - Don't expose sensitive data
  - Respect blocks/reports
  - GDPR compliance (can delete user data)
- **Challenge**: Store embeddings without raw data exposure

### Success Metrics

#### User Engagement
- **Daily Active Users (DAU)**
- **Session duration** (time spent swiping)
- **Swipes per session** (engagement with recommendations)

#### Recommendation Quality
- **Like rate** (% of recommendations that get liked) - Target: >10%
- **Match rate** (% of likes that become mutual matches) - Target: >5%
- **Conversation rate** (% of matches that lead to messages) - Target: >30%

#### Business Metrics
- **User retention** (7-day, 30-day)
- **Time to first match** (cold start effectiveness)
- **Revenue** (if premium features: super likes, boosts)

#### ML System Metrics
- **Recommendation latency** (p50, p95, p99)
- **Model prediction accuracy** (offline metrics)
- **Feature computation time**
- **Cache hit rate**

### Key Constraints

#### Scale Constraints
- **Database**: Must handle millions of users, billions of interactions
- **Computation**: Can't score every user pair in real-time
- **Storage**: User embeddings, interaction history, features

#### Real-Time Constraints
- **Latency budget**: 300ms end-to-end
- **Breakdown**:
  - Database query: 50ms
  - Candidate generation: 100ms
  - Feature computation: 50ms
  - Ranking inference: 50ms
  - API overhead: 50ms

#### Business Constraints
- **Cost**: Inference cost per recommendation request
- **Infrastructure**: Must run on commodity hardware (no expensive GPUs initially)
- **Development velocity**: Need to iterate and A/B test quickly

### Problem Space Summary

**Core Challenge**:
Generate personalized, real-time recommendations for millions of users with <300ms latency, balancing relevance, diversity, and fairness in a two-sided marketplace.

**Key Insights**:
1. **Scale problem**: Can't score everyone → need efficient retrieval
2. **Speed problem**: Must be fast → need caching, optimization
3. **Accuracy problem**: Must be relevant → need ML
4. **Fairness problem**: Must balance exploitation/exploration
5. **Cold start problem**: Must blend collaborative + content-based

**This problem naturally suggests a two-stage approach**:
- **Stage 1 (Retrieval)**: Quickly narrow down millions of users to 100-500 candidates
- **Stage 2 (Ranking)**: Accurately score/rank those candidates with expensive ML

---

## Interview Narrative (Phase 1)

**"Tell me about your recommender system"**

> "I built a two-stage recommendation system for a dating app. The core problem was: how do you generate personalized matches for millions of users in under 300ms?
>
> Dating recommendations are unique because they require mutual interest - it's a two-sided marketplace. You can't just optimize for one user's preferences; you have to consider whether the other person would also be interested.
>
> The key constraint was latency. Users expect an instant feed when they open the app. But we can't afford to score millions of users individually in real-time - that would take seconds or minutes.
>
> This naturally led to a two-stage design: a fast retrieval stage to narrow down candidates, followed by an accurate ranking stage using ML. I'll walk you through each stage..."

**Key Trade-offs to Articulate**:
1. **Latency vs Accuracy**: Fast approximate retrieval vs expensive accurate ranking
2. **Exploration vs Exploitation**: Show their "type" vs introduce diversity
3. **Collaborative vs Content-based**: Use past behavior vs profile features
4. **Real-time vs Batch**: Fast updates vs computational efficiency

---

## Common Follow-up Questions & Crisp Answers

### Q: "Why not just use SQL queries/rules instead of ML?"

**Strong Answer (30 seconds):**

> "SQL with filters can handle basic constraints like age range and location, but it can't personalize based on subtle preferences. If User A likes profiles with 'hiking' and 'photography' interests, a rule can match those keywords - but it won't learn that User A prefers certain bio writing styles, or that they tend to like users who've been active recently.
>
> ML learns patterns from interaction history that we can't hard-code. However, you're right that simple is fast - that's why our Stage 1 uses efficient approximate retrieval methods like FAISS, not expensive neural networks. We save the ML complexity for Stage 2 where we only score 100-500 candidates, not millions."

**Key Points Hit:**
- ✓ SQL handles explicit filters
- ✓ ML learns implicit preferences from behavior
- ✓ Acknowledge interviewer's valid point (simple = fast)
- ✓ Connect to two-stage design

---

### Q: "How is dating different from Netflix? Why not standard collaborative filtering?"

**Strong Answer (45 seconds):**

> "Three key differences:
>
> **1. Two-sided marketplace**: Netflix recommends movies TO you - the movie doesn't care if you watch it. Dating requires mutual interest. If I recommend User A to User B, User B also needs to find User A attractive. Standard CF only optimizes one direction.
>
> **2. Fairness problem**: With standard CF, popular users (like popular movies) get recommended to everyone. In dating, that creates a terrible experience - attractive users get overwhelmed, average users get ignored. We need exploration and fairness constraints.
>
> **3. Real-time feedback**: Netflix can batch-update recommendations overnight. Dating needs to incorporate your swipes immediately - if you liked 5 brunettes, the next feed should reflect that preference right away."

**Key Points Hit:**
- ✓ Mutual interest (two-sided)
- ✓ Fairness/popularity bias
- ✓ Real-time feedback loop
- ✓ Shows you understand domain-specific constraints

---

### Q: "What's your target latency and why?"

**Crisp Answer (20 seconds):**

> "300ms p95. Why? User opens app expecting an instant feed - anything over 500ms feels slow. We budget 100ms for candidate generation, 50ms each for DB queries, feature computation, and ranking, with 50ms for API overhead. This constraint drives our two-stage design - we can't afford to ML-score millions of users in real-time."

---

### Q: "How do you measure success?"

**Crisp Answer (30 seconds):**

> "Three layers of metrics:
>
> **User engagement**: Like rate (target >10%), match rate (>5%), conversation rate (>30%)
>
> **Business**: 7-day and 30-day retention, time to first match
>
> **System**: Recommendation latency p95 <300ms, model prediction accuracy, cache hit rate
>
> The key metric is match rate - we want recommendations that lead to mutual interest, not just one-sided likes."

---

## Next Steps (Still in Phase 1)
- [ ] Validate this problem understanding against codebase
- [ ] Check if there are explicit requirement docs
- [ ] Look for any product specs or feature requirements
- [ ] Move to Phase 2: Understand how architecture addresses these requirements
