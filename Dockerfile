
FROM python:3.13.0

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

# Create static directory for any generated files
RUN mkdir -p /code/static && chmod 777 /code/static

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app

# Create health check endpoint file
RUN mkdir -p /code/app/static && \
    echo '{"status": "healthy"}' > /code/app/static/health.json

# Expose ports
EXPOSE 80
EXPOSE 9090

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]