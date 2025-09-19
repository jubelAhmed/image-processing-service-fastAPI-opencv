# Facial Contour Masking API

A modern, enterprise-grade FastAPI application for processing facial images and generating SVG contour masks for specific facial regions.

## 🚀 Features

### Core Functionality
- **Facial Image Processing**: Accepts facial images with landmarks and segmentation maps
- **SVG Generation**: Returns SVG contour masks for specific facial regions
- **Face Alignment**: Handles autorotation and cropping of faces
- **Asynchronous Processing**: Uses background tasks for non-blocking API calls

### Enterprise Features
- **🔐 JWT Authentication**: Secure user authentication with access/refresh tokens
- **🛡️ Security Middleware**: CORS, security headers, and request logging
- **⚡ Rate Limiting**: Redis-backed rate limiting with `slowapi`
- **📊 Monitoring**: Prometheus metrics and Grafana dashboards
- **🗄️ Database Management**: SQLAlchemy ORM with Alembic migrations
- **💾 Perceptual Caching**: Smart caching using image similarity hashing
- **🎨 Rich Logging**: Beautiful console output with structured logging

## 🏗️ Architecture

```
┌─────────────┐     ┌───────────────┐     ┌───────────────┐
│ HTTP Client │────▶│ FastAPI       │────▶│ Background    │
│             │◀────│ Endpoint      │     │ Task Worker   │
└─────────────┘     └───────────────┘     └───────┬───────┘
                           │                      │
                           ▼                      ▼
                    ┌───────────────┐     ┌───────────────┐
                    │ PostgreSQL DB │◀────│ Image         │
                    │ (Job Status)  │     │ Processor     │
                    └───────────────┘     └───────────────┘
                           ▲                      │
                           │                      ▼
                    ┌───────────────┐     ┌───────────────┐
                    │ Perceptual    │◀────│ SVG Generator │
                    │ Hash Cache    │     │               │
                    └───────────────┘     └───────────────┘
```

## 📁 Project Structure

```
opnecv_image_processing/
├── .venv/                    # uv virtual environment
├── src/                      # Source code
│   ├── auth/                 # Authentication module
│   │   ├── router.py         # Auth endpoints
│   │   ├── models.py         # User & token models
│   │   ├── schemas.py        # Pydantic schemas
│   │   ├── service.py        # Business logic
│   │   ├── dependencies.py   # FastAPI dependencies
│   │   └── security.py       # JWT & password handling
│   ├── facial/               # Facial processing module
│   │   ├── router.py         # Processing endpoints
│   │   ├── models.py         # Processing models
│   │   ├── schemas.py        # Request/response schemas
│   │   ├── service.py        # Processing logic
│   │   ├── image_generator.py # Image generation
│   │   ├── generators/       # Output generators (SVG, PNG, JSON)
│   │   └── facial_processing/ # Core processing
│   ├── core/                 # Global utilities
│   │   ├── config.py         # Configuration
│   │   ├── database.py       # Database connection
│   │   ├── models.py         # Base models
│   │   └── utils.py          # Utilities
│   ├── middleware/           # Custom middleware
│   │   ├── rate_limiting.py  # Rate limiting
│   │   └── security.py       # Security headers
│   ├── monitoring/           # Monitoring setup
│   │   └── prometheus.py     # Prometheus metrics
│   └── main.py              # Application entry point
├── requirements/             # Dependencies
│   ├── base.txt             # Core dependencies
│   ├── dev.txt              # Development tools
│   └── prod.txt             # Production dependencies
├── alembic/                 # Database migrations
├── docker/                  # Docker configurations
└── templates/               # Static files & dashboards
```

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- Docker & Docker Compose (optional)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd opnecv_image_processing
   ```

2. **Set up virtual environment with uv**
   ```bash
   # Install uv if not already installed
   pip install uv
   
   # Create virtual environment
   uv venv
   
   # Activate virtual environment
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   uv pip install -r requirements/base.txt
   ```

4. **Run the application**
   ```bash
   # Development mode (without database)
   DB_USE_DATABASE=false fastapi run src.main:app --reload
   
   # Or with uvicorn
   DB_USE_DATABASE=false uvicorn src.main:app --reload
   ```

5. **Access the application**
   - API: http://localhost:8000
   - Documentation: http://localhost:8000/docs
   - Prometheus: http://localhost:9090

## 🐳 Docker Deployment

### Using Docker Compose
```bash
# Start all services
docker-compose up --build

# Run in background
docker-compose up -d
```

### Services
- **API**: http://localhost:8000
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

## 🔧 Configuration

### Environment Variables

#### Application Settings
- `APP_NAME`: Application name (default: "Facial Contour Masking API")
- `APP_VERSION`: Application version (default: "1.0.0")
- `APP_DEBUG`: Enable debug mode (default: false)

#### Database Settings
- `DB_USE_DATABASE`: Enable/disable database (default: true)
- `DATABASE_URL`: Database connection string
  - PostgreSQL: `postgresql://user:pass@host:port/db`
  - SQLite: `sqlite:///./facial_api.db`

#### Authentication Settings
- `AUTH_SECRET_KEY`: JWT secret key (change in production!)
- `AUTH_ACCESS_TOKEN_EXPIRE_MINUTES`: Access token expiration (default: 30)
- `AUTH_REFRESH_TOKEN_EXPIRE_DAYS`: Refresh token expiration (default: 7)

#### Rate Limiting Settings
- `REDIS_URL`: Redis connection string (default: redis://localhost:6379)

#### Monitoring Settings
- `PROMETHEUS_ENABLED`: Enable Prometheus metrics (default: true)
- `PROMETHEUS_PORT`: Prometheus port (default: 9090)

## 📚 API Documentation

### Authentication Endpoints

#### POST /api/v1/auth/register
Register a new user
```json
{
  "username": "user@example.com",
  "password": "secure_password",
  "full_name": "John Doe"
}
```

#### POST /api/v1/auth/login
Login and get tokens
```json
{
  "username": "user@example.com",
  "password": "secure_password"
}
```

### Processing Endpoints

#### POST /api/v1/frontal/crop/submit
Submit facial image for processing
```json
{
  "image": "base64_encoded_image",
  "landmarks": [{"x": 0, "y": 0}, ...],
  "segmentation_map": "base64_encoded_segmentation_map"
}
```

#### GET /api/v1/frontal/crop/status/{job_id}
Check processing status
```json
{
  "job_id": "uuid",
  "status": "completed",
  "result": "base64_encoded_svg"
}
```

## 🗄️ Database Management

### Using Alembic Migrations
```bash
# Create a new migration
alembic revision --autogenerate -m "Description"

# Apply migrations
alembic upgrade head

# Rollback migration
alembic downgrade -1
```

### Database Models
- **Users**: User authentication and profiles
- **RefreshTokens**: JWT refresh token management
- **Cache**: Perceptual hash caching
- **Jobs**: Processing job status
- **ProcessingMetrics**: Performance metrics

## 🛠️ Development

### Adding Dependencies
```bash
# Using uv (recommended)
uv pip install package-name

# Add to requirements
uv pip freeze > requirements/base.txt
```

### Code Quality
```bash
# Install development dependencies
uv pip install -r requirements/dev.txt

# Run tests (when implemented)
pytest

# Format code
black src/
isort src/
```

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src
```

## 🔒 Security Features

- **JWT Authentication**: Secure token-based authentication
- **Password Hashing**: Bcrypt password hashing
- **Rate Limiting**: Redis-backed rate limiting
- **CORS Protection**: Configurable CORS policies
- **Security Headers**: XSS, CSRF, and content type protection
- **Request Logging**: Comprehensive request/response logging

## 📊 Monitoring & Observability

### Prometheus Metrics
- API request counts and latency
- Image processing time
- Job status distribution
- Authentication events
- Rate limiting violations

### Grafana Dashboards
- Real-time API performance
- Error rates and response times
- User activity and authentication
- System resource usage

## 🚀 Production Deployment

### Environment Setup
1. Set production environment variables
2. Use PostgreSQL for database
3. Configure Redis for rate limiting
4. Set up proper logging
5. Configure monitoring

### Performance Optimization
- Database connection pooling
- Redis caching
- Async processing
- Prometheus monitoring
- Rate limiting protection

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Check the API documentation at `/docs`
- Review the monitoring dashboards

---

