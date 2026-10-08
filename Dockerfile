# Production Dockerfile for Bank Churn Decision System
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency requirements
COPY requirements.txt pyproject.toml README.md ./

# Install runtime dependencies and churn package
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -e .

# Copy application and model artifacts
COPY src/ src/
COPY models/ models/
COPY reports/ reports/
COPY app/ app/

EXPOSE 8000 8501

# Default: Launch FastAPI server
CMD ["uvicorn", "churn.api:app", "--host", "0.0.0.0", "--port", "8000"]
