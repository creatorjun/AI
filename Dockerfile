# rag_service/Dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .

RUN pip install --no-cache-dir \
    "fastapi>=0.111.0" \
    "uvicorn[standard]>=0.29.0" \
    "sqlalchemy[asyncio]>=2.0.0" \
    "asyncpg>=0.29.0" \
    "pgvector>=0.3.0" \
    "alembic>=1.13.0" \
    "pydantic-settings>=2.2.0" \
    "openai>=1.30.0" \
    "httpx>=0.27.0" \
    "watchdog>=4.0.0" \
    "pytest>=8.0.0" \
    "pytest-asyncio>=0.23.0"

COPY . .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
