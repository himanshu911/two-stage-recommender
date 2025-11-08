# Two-Stage Recommender System

A production-ready dating recommender system built with FastAPI, featuring a two-stage recommendation pipeline with collaborative filtering and machine learning-based ranking.

## 🚀 Features

- **Two-Stage Recommendation Pipeline**
  - Candidate Generation: Collaborative filtering + content-based filtering + exploration
  - Ranking: Logistic regression model with real-time features
- **Machine Learning Integration**
  - Matrix factorization for user embeddings
  - FAISS for efficient similarity search
  - Feature store for consistent ML features
- **Production-Ready Architecture**
  - Repository pattern for data access
  - Dependency injection for testability
  - Comprehensive error handling and logging
  - Health checks and monitoring
- **RESTful API**
  - User management (CRUD operations)
  - Interaction tracking (likes, dislikes, super likes)
  - Personalized recommendations with explanations
  - Comprehensive filtering and pagination

## 🏗️ Architecture

The system follows a microservices-inspired architecture with clean separation of concerns:

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
│   PostgreSQL    │  In-Memory Cache│   Feature Store (PostgreSQL)│
│   (Primary DB)  │  (Redis ready)  │                             │
└─────────────────┴─────────────────┴─────────────────────────────┘
```

**Note**: Current implementation uses in-memory caching. Redis is available in docker-compose for production deployment.

## 🛠️ Technology Stack

- **Web Framework**: FastAPI with async support
- **Database**: PostgreSQL with SQLModel ORM
- **Cache**: In-memory caching (Redis available in docker-compose for production use)
- **ML Framework**: Scikit-learn, PyTorch, FAISS
- **Testing**: pytest with async support
- **Deployment**: Docker with multi-stage builds
- **Monitoring**: Structured logging, health checks, metrics

## 📦 Installation

### Prerequisites

- Python 3.11+
- PostgreSQL 15+
- Redis 7+ (optional, included in docker-compose)
- Docker (optional, for containerized deployment)

### Local Development Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd two-stage-recommender
   ```

2. **Create virtual environment**
   ```bash
   # Using conda (recommended)
   conda env create -f environment.yml
   conda activate two-stage-recommender
   
   # Or using pip
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   # Create .env file with your configuration
   cat > .env << EOF
   DATABASE_URL=postgresql://postgres:postgres@localhost:5432/two-stage-recommender
   REDIS_URL=redis://localhost:6379
   ENVIRONMENT=development
   DEBUG=True
   LOG_LEVEL=INFO
   EOF
   ```

4. **Set up database**
   ```bash
   # Create database
   createdb two-stage-recommender

   # (Optional) Add PostgreSQL extensions
   psql two-stage-recommender < scripts/init-db.sql

   # Tables will be created automatically on application startup via SQLModel
   ```

5. **Run the application**
   ```bash
   # Development mode with auto-reload
   uvicorn app.main:app --reload
   
   # Or using the run script
   python -m app.main
   ```

### Docker Setup

1. **Using Docker Compose**
   ```bash
   # Start all services (app, PostgreSQL, Redis)
   docker-compose up -d
   
   # View logs
   docker-compose logs -f app
   
   # Stop services
   docker-compose down
   ```

2. **Manual Docker build**
   ```bash
   # Build the image
   docker build -t two-stage-recommender .
   
   # Run the container
   docker run -p 8000:8000 --env-file .env two-stage-recommender
   ```

## 📚 API Documentation

### Interactive API Documentation

Once the application is running, access the interactive API documentation:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

> **Note**: Authentication is not currently implemented. All API endpoints are publicly accessible.

### Endpoints

#### User Management
- `POST /api/v1/users/` - Create a new user
- `GET /api/v1/users/{user_id}` - Get user by ID
- `PUT /api/v1/users/{user_id}` - Update user
- `DELETE /api/v1/users/{user_id}` - Delete user
- `GET /api/v1/users/` - List users with filtering (location, age range, pagination)
- `GET /api/v1/users/search/active` - Search for active users
- `GET /api/v1/users/search/by-interest` - Search users by interest

#### Interactions
- `POST /api/v1/interactions/` - Create interaction (like/dislike/super_like/block)
- `GET /api/v1/interactions/user/{user_id}` - Get user interactions (with filtering)
- `GET /api/v1/interactions/stats/{user_id}` - Get interaction statistics
- `GET /api/v1/interactions/mutual/{user_id}/{target_user_id}` - Get mutual interactions
- `GET /api/v1/interactions/recent/{user_id}` - Get recent interactions
- `GET /api/v1/interactions/timeline/{user_id}` - Get interaction timeline

#### Recommendations
- `GET /api/v1/recommendations/users/{user_id}/recommendations` - Get personalized recommendations
- `GET /api/v1/recommendations/users/{user_id}/recommendations/explain` - Get recommendations with explanations
- `GET /api/v1/recommendations/users/{user_id}/similar` - Find similar users
- `POST /api/v1/recommendations/refresh` - Refresh recommendation cache
- `GET /api/v1/recommendations/performance` - Get recommendation performance metrics
- `GET /api/v1/recommendations/algorithm/versions` - Get available algorithm versions

#### Health & Monitoring
- `GET /api/v1/health/` - Comprehensive health check
- `GET /api/v1/health/ready` - Readiness check (for Kubernetes/load balancer)
- `GET /api/v1/health/live` - Liveness check
- `GET /api/v1/health/metrics` - Application metrics
- `GET /api/v1/health/models` - ML model information
- `GET /api/v1/health/dependencies` - Dependency status information

### Example Usage

#### Create a User
```bash
curl -X POST "http://localhost:8000/api/v1/users/" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Alice",
    "age": 25,
    "gender": "female",
    "location": "San Francisco",
    "bio": "Love hiking and photography",
    "interests": ["hiking", "photography", "travel"]
  }'
```

#### Get Recommendations
```bash
curl "http://localhost:8000/api/v1/recommendations/users/1/recommendations?limit=10"
```

#### Track Interaction
```bash
curl -X POST "http://localhost:8000/api/v1/interactions/" \
  -H "Content-Type: application/json" \
  -d '{
    "target_user_id": 2,
    "interaction_type": "like",
    "context": {"source": "recommendation"}
  }'
```

## 🧪 Testing

### Run Tests

```bash
# Run all tests
pytest

# Run specific test types
pytest -m unit
pytest -m integration
pytest -m api

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/unit/test_user_repository.py

# Run with detailed output
pytest -v
```

### Test Structure

```
tests/
├── unit/           # Unit tests (fast, isolated)
│   ├── test_user_repository.py
│   ├── test_feature_service.py
│   └── ...
├── integration/    # Integration tests (with database)
│   ├── test_user_api.py
│   ├── test_recommendation_api.py
│   └── ...
└── conftest.py    # Test configuration and fixtures
```

**Note**: End-to-end tests are planned for future implementation.

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://user:password@localhost:5432/two-stage-recommender` |
| `REDIS_URL` | Redis connection string | `redis://localhost:6379` |
| `ENVIRONMENT` | Environment (development/production) | `development` |
| `DEBUG` | Enable debug mode | `False` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `EMBEDDING_DIMENSION` | User embedding dimension | `64` |
| `CANDIDATE_GENERATION_TOP_K` | Max candidates to generate | `100` |

### Configuration Files

- `.env` - Environment variables
- `pytest.ini` - Test configuration with markers and coverage
- `docker-compose.yml` - Docker services configuration
- `environment.yml` - Conda environment specification
- `requirements.txt` - Python package dependencies

## 📈 Monitoring and Observability

### Health Checks

The application provides comprehensive health checks:

```bash
# Overall health
curl http://localhost:8000/api/v1/health/

# Readiness check
curl http://localhost:8000/api/v1/health/ready

# Performance metrics
curl http://localhost:8000/api/v1/health/metrics
```

### Logging

The application uses structured logging with different levels:
- **DEBUG**: Detailed debugging information
- **INFO**: General application flow
- **WARNING**: Potential issues
- **ERROR**: Error conditions
- **CRITICAL**: Critical errors

### Performance Metrics

Key metrics tracked:
- Recommendation generation time
- Database query performance
- Cache hit rates
- Model inference time
- API response times

## 🚀 Deployment

### Production Deployment

1. **Environment Setup**
   ```bash
   # Set production environment variables
   export ENVIRONMENT=production
   export DEBUG=false
   export LOG_LEVEL=WARNING
   ```

2. **Database Setup**
   ```bash
   # Tables are created automatically via SQLModel on startup
   # Optionally add PostgreSQL extensions:
   psql $DATABASE_URL < scripts/init-db.sql
   ```

3. **Start Application**
   ```bash
   # Using gunicorn for production
   gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
   ```

4. **Load Balancer Configuration**
   ```nginx
   upstream app {
       server localhost:8000;
       server localhost:8001;
       server localhost:8002;
       server localhost:8003;
   }
   
   server {
       listen 80;
       location / {
           proxy_pass http://app;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

### CI/CD Pipeline

Example GitHub Actions workflow:

```yaml
name: Deploy
on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.11
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      - name: Run tests
        run: pytest --cov=app

  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to production
        run: |
          # Deployment commands
```

## 🔮 Future Enhancements

### Planned Features

1. **Authentication & Authorization**
   - JWT token-based authentication
   - User registration and login
   - Role-based access control (RBAC)
   - API rate limiting per user

2. **Real-time ML Updates**
   - Incremental model training
   - Real-time feature updates
   - Model drift detection

3. **Advanced Recommendation Strategies**
   - Deep learning models
   - Multi-objective optimization
   - Context-aware recommendations

4. **Social Features**
   - Friend recommendations
   - Group matching
   - Event-based recommendations

5. **Enhanced Privacy**
   - Differential privacy
   - Federated learning
   - Enhanced data encryption

### Performance Improvements

1. **Caching Strategy**
   - Redis cluster for high availability
   - CDN for static content
   - Application-level caching

2. **Database Optimization**
   - Read replicas for scaling
   - Connection pooling optimization
   - Query optimization

3. **ML Optimization**
   - Model quantization
   - GPU inference acceleration
   - Batch prediction optimization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add tests for new features
- Update documentation
- Use type hints
- Write descriptive commit messages

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- FastAPI community for the excellent framework
- SQLModel for the type-safe ORM
- Scikit-learn team for ML algorithms
- PostgreSQL and Redis communities for robust databases

## 📞 Support

For questions or support:
- Create an issue in the repository
- Check the documentation in `/docs`
- Review the architecture guide in `ARCHITECTURE.md`

---

**Built with ❤️ for better human connections**
