# app/main.py
from fastapi import FastAPI, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Annotated

from app.api.routers import ingest, manage, search
from app.config import settings
from app.database import get_db

app = FastAPI(title="RAG Service", version="1.0.0")

app.include_router(ingest.router)
app.include_router(search.router)
app.include_router(manage.router)


@app.get("/health")
async def health(session: Annotated[AsyncSession, Depends(get_db)]) -> dict:
    await session.execute(text("SELECT 1"))
    return {
        "status": "ok",
        "env": settings.app_env,
        "db": "connected",
    }