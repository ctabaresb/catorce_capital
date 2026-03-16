# =============================================================================
# Dockerfile
# Base image for ECS Fargate tasks:
#   - Historical backfill (Part 5)
#   - Backtest engine (Week 2)
#   - GBM simulation engine (Week 3)
#
# Build: docker build -t crypto-platform-engine .
# Run locally:
#   docker run --rm \
#     -e DATA_LAKE_BUCKET=crypto-platform-catorce \
#     -e COINGECKO_API_KEY=your-key \
#     -e AWS_DEFAULT_REGION=us-east-1 \
#     -v ~/.aws:/root/.aws:ro \
#     crypto-platform-engine \
#     python -m ingestion.backfill --start-date 2024-01-01 --max-assets 5
# =============================================================================

FROM python:3.12-slim

# ---------------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Working directory
# ---------------------------------------------------------------------------
WORKDIR /app

# ---------------------------------------------------------------------------
# Python dependencies
# Install before copying src/ so Docker layer cache is used on code changes
# ---------------------------------------------------------------------------
COPY requirements-ecs.txt .
RUN pip install --no-cache-dir -r requirements-ecs.txt

# ---------------------------------------------------------------------------
# Application source
# ---------------------------------------------------------------------------
COPY src/ ./src/

# ---------------------------------------------------------------------------
# Python path: makes 'from ingestion.x import' work without installation
# ---------------------------------------------------------------------------
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# ---------------------------------------------------------------------------
# Default command: show help (overridden by ECS task definition)
# ---------------------------------------------------------------------------
CMD ["python", "-m", "ingestion.backfill", "--help"]
