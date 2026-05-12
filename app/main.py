# app/main.py
from fastapi import FastAPI
from sqlalchemy import text

from app.api.routers import ingest, manage, search
from app.config import settings
from app.database import AsyncSessionLocal

app = FastAPI(title="RAG Service", version="1.0.0")

app.include_router(ingest.router)
app.include_router(search.router)
app.include_router(manage.router)


@app.get("/health")
async def health() -> dict:
    async with AsyncSessionLocal() as session:
        await session.execute(text("SELECT 1"))
    return {
        "status": "ok",
        "env": settings.app_env,
        "db": "connected",
    }