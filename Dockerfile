FROM python:3.11-slim

WORKDIR /app

# Postgres build deps (safe even if using psycopg2-binary)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev \
 && rm -rf /var/lib/apt/lists/*

# Install deps first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

ENV PYTHONUNBUFFERED=1

# DO probes $PORT; default 8080 locally
EXPOSE 8080

# ✅ Start command: Gunicorn with Uvicorn workers, binding to $PORT
CMD ["bash", "-lc", "gunicorn -k uvicorn.workers.UvicornWorker -w 2 -b 0.0.0.0:${PORT:-8080} main:app"]
