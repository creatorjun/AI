# app/api/deps.py
from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.application.ingest_usecase import IngestUsecase
from app.application.manage_usecase import ManageUsecase
from app.application.search_usecase import SearchUsecase
from app.config import settings
from app.database import get_db
from app.domain.ports import IReranker
from app.infrastructure.chunker import SentenceChunker
from app.infrastructure.openai_embedder import OpenAIEmbedder
from app.infrastructure.pg_vector_store import PgVectorStore
from app.infrastructure.vllm_reranker import NoOpReranker, VLLMReranker

_embedder = OpenAIEmbedder()
_reranker: IReranker = VLLMReranker() if settings.reranker_enabled else NoOpReranker()
_chunker = SentenceChunker()


async def get_ingest_usecase(
    session: Annotated[AsyncSession, Depends(get_db)],
) -> IngestUsecase:
    return IngestUsecase(
        embedder=_embedder,
        chunker=_chunker,
        vector_store=PgVectorStore(session),
    )


async def get_manage_usecase(
    session: Annotated[AsyncSession, Depends(get_db)],
) -> ManageUsecase:
    return ManageUsecase(
        embedder=_embedder,
        vector_store=PgVectorStore(session),
    )


async def get_search_usecase(
    session: Annotated[AsyncSession, Depends(get_db)],
) -> SearchUsecase:
    return SearchUsecase(
        embedder=_embedder,
        vector_store=PgVectorStore(session),
        reranker=_reranker,
    )


IngestDep = Annotated[IngestUsecase, Depends(get_ingest_usecase)]
ManageDep = Annotated[ManageUsecase, Depends(get_manage_usecase)]
SearchDep = Annotated[SearchUsecase, Depends(get_search_usecase)]