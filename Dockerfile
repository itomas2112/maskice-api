# Use slim Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for psycopg2 (Postgres driver) and build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements first (to leverage Docker layer caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Ensure Python output is not buffered
ENV PYTHONUNBUFFERED=1

# DigitalOcean passes $PORT automatically; default to 8080 if not set
EXPOSE 8080

# Start app using Gunicorn with Uvicorn workers (production-ready)
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "2", "-b", "0.0.0.0:${PORT:-8080}", "main:app"]
