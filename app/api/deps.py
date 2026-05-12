# app/api/deps.py
from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.application.ingest_usecase import IngestUsecase
from app.application.manage_usecase import ManageUsecase
from app.application.search_usecase import SearchUsecase
from app.config import settings
from app.database import get_db
from app.infrastructure.chunker import SentenceChunker
from app.infrastructure.openai_embedder import OpenAIEmbedder
from app.infrastructure.pg_vector_store import PgVectorStore
from app.infrastructure.vllm_reranker import NoOpReranker, VLLMReranker
from app.domain.ports import IReranker


def get_embedder() -> OpenAIEmbedder:
    return OpenAIEmbedder()


def get_reranker() -> IReranker:
    if settings.reranker_enabled:
        return VLLMReranker()
    return NoOpReranker()


async def get_ingest_usecase(
    session: Annotated[AsyncSession, Depends(get_db)],
) -> IngestUsecase:
    return IngestUsecase(
        embedder=get_embedder(),
        chunker=SentenceChunker(),
        vector_store=PgVectorStore(session),
    )


async def get_manage_usecase(
    session: Annotated[AsyncSession, Depends(get_db)],
) -> ManageUsecase:
    return ManageUsecase(
        embedder=get_embedder(),
        vector_store=PgVectorStore(session),
    )


async def get_search_usecase(
    session: Annotated[AsyncSession, Depends(get_db)],
) -> SearchUsecase:
    return SearchUsecase(
        embedder=get_embedder(),
        vector_store=PgVectorStore(session),
        reranker=get_reranker(),
    )


IngestDep = Annotated[IngestUsecase, Depends(get_ingest_usecase)]
ManageDep = Annotated[ManageUsecase, Depends(get_manage_usecase)]
SearchDep = Annotated[SearchUsecase, Depends(get_search_usecase)]