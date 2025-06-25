# Facial Contour Masking API

This FastAPI application processes facial images and generates SVG contour masks for specific facial regions.

## Features

- Accepts facial images with landmarks and segmentation map
- Processes the image to identify facial regions
- Returns an SVG with contour masks for facial regions
- Handles autorotation and cropping of faces

### Advanced Features

- **Asynchronous Processing**: Uses background tasks for non-blocking API calls
- **Prometheus Monitoring**: Tracks API requests, processing time, and job status
- **Rich Console Logs**: Beautiful and informative terminal output
- **Performance Optimization**: Parallel processing and optimized algorithms
- **PostgreSQL Caching**: Stores and retrieves results for similar requests using perceptual hashing
- **Configurable Database Usage**: Can run with or without database connection

## Architecture

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


## Getting Started

### Prerequisites

- Docker and Docker Compose

### Installation

1. Clone this repository
2. Navigate to the project directory
3. Run the application using Docker Compose:

```bash
docker-compose up --build
```

The API will be available at `http://localhost:8000`.
Prometheus metrics are available at `http://localhost:9090`.
Grafana dashboard is available at `http://localhost:3000`.

## API Endpoints

### POST /api/v1/frontal/crop/submit

Submits a facial image for processing and returns a job ID.

#### Request Payload

```json
{
  "image": "base64_encoded_image",
  "landmarks": [{"x": 0, "y": 0}, ...],
  "segmentation_map": "base64_encoded_segmentation_map",
  "loadtest_mode": false
}
```

Set `loadtest_mode` to `true` to bypass the simulated processing delay.

#### Response

```json
{
  "id": "job_id",
  "status": "pending"
}
```

### GET /api/v1/frontal/crop/status/{job_id}

Checks the status of a submitted job.

#### Response

```json
{
  // "id": "job_id",
  // "status": "pending|processing|completed|failed",
  "svg": "base64_encoded_svg",
  "mask_contours": {"1": [...], "2": [...], ...}
  
}
```

## Monitoring and Observability

### Prometheus Metrics

The application exposes Prometheus metrics at `http://localhost:9090`. These include:

- API request counts
- Request latency
- Image processing time
- Job status counts

### Grafana Dashboard

A Grafana dashboard is available at `http://localhost:3000`. Default credentials are:

- Username: admin
- Password: admin

## Development

### Local Development

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   fastapi dev app/main.py
   ```

```

## Configuration

The application can be configured using environment variables:

### Application Settings
- `APP_NAME`: Name of the application (default: "Facial Contour Masking API")
- `APP_NAME`: Name of the application (default: "Facial Contour Masking API")
- `APP_VERSION`: Application version (default: "1.0.0")
- `APP_DEBUG`: Enable debug mode (default: false)

### Database Settings
- `DB_USE_DATABASE`: Enable/disable database usage (default: true)
- `DB_HOST`: PostgreSQL host (default: "postgres")
- `DB_PORT`: PostgreSQL port (default: 5432)
- `DB_USERNAME`: PostgreSQL username (default: "postgres")
- `DB_PASSWORD`: PostgreSQL password (default: "postgres")
- `DB_DATABASE`: PostgreSQL database name (default: "facial_api")

### Prometheus Settings
- `PROMETHEUS_ENABLED`: Enable/disable Prometheus metrics (default: true)
- `PROMETHEUS_PORT`: Port for Prometheus metrics server (default: 9090)

## Running Without Database

To run the application without database functionality:

```bash
# Set the environment variable
export DB_USE_DATABASE=false

# Run the application
fastapi dev app/main.py
```

Or with Docker Compose:

```bash
# Edit the docker-compose.yml file to set DB_USE_DATABASE=false
docker-compose up -d
```

## Architecture

The application follows a modular design with the following components:

- **API Layer**: FastAPI routes and request handling
- **Job Queue**: Asynchronous job processing
- **Image Processing**: Core image analysis and SVG generation
- **Caching**: PostgreSQL-based result caching (optional)
- **Monitoring**: Prometheus metrics and Rich logging

This architecture allows for easy extension and scaling of the application.
