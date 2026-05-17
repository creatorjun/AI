# app/main.py
import asyncio
import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import Depends, FastAPI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.routers import ingest, manage, search, generate
from app.api.deps import _embedder, _chunker
from app.application.ingest_usecase import IngestUsecase
from app.config import settings
from app.database import AsyncSessionLocal, get_db
from app.infrastructure.folder_watcher import FolderWatcher
from app.infrastructure.pg_vector_store import PgVectorStore

logger = logging.getLogger(__name__)

_folder_watcher: FolderWatcher | None = None


def _make_ingest_usecase() -> IngestUsecase:
    session = AsyncSessionLocal()
    return IngestUsecase(
        embedder=_embedder,
        chunker=_chunker,
        vector_store=PgVectorStore(session),
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global _folder_watcher
    if settings.watch_folder:
        usecase = _make_ingest_usecase()
        _folder_watcher = FolderWatcher(
            folder_path=settings.watch_folder,
            usecase=usecase,
        )
        _folder_watcher.start(asyncio.get_event_loop())
    yield
    if _folder_watcher:
        _folder_watcher.stop()


app = FastAPI(title="RAG Service", version="1.2.0", lifespan=lifespan)

app.include_router(ingest.router)
app.include_router(search.router)
app.include_router(manage.router)
app.include_router(generate.router)


@app.get("/health")
async def health(session: Annotated[AsyncSession, Depends(get_db)]) -> dict:
    await session.execute(text("SELECT 1"))
    return {
        "status": "ok",
        "env": settings.app_env,
        "db": "connected",
        "embedding_backend": settings.embedding_backend,
        "llm_backend": settings.llm_backend,
        "watch_folder": settings.watch_folder or None,
    }
