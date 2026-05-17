# app/api/deps.py
from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.application.generate_usecase import GenerateUsecase
from app.application.ingest_usecase import IngestUsecase
from app.application.manage_usecase import ManageUsecase
from app.application.search_usecase import SearchUsecase
from app.config import settings
from app.database import get_db
from app.domain.ports import IEmbedder, IGenerator, IReranker
from app.infrastructure.chunker import SemanticChunker
from app.infrastructure.openai_embedder import OpenAIEmbedder
from app.infrastructure.pg_vector_store import PgVectorStore
from app.infrastructure.vllm_reranker import NoOpReranker, VLLMReranker


def _build_embedder() -> IEmbedder:
    if settings.embedding_backend == "ollama":
        from app.infrastructure.ollama_embedder import OllamaEmbedder
        return OllamaEmbedder()
    return OpenAIEmbedder()


def _build_reranker() -> IReranker:
    if not settings.reranker_enabled:
        return NoOpReranker()
    if settings.llm_backend == "mlx":
        from app.infrastructure.mlx_reranker import MLXReranker
        return MLXReranker()
    if settings.llm_backend == "ollama":
        from app.infrastructure.ollama_reranker import OllamaReranker
        return OllamaReranker()
    return VLLMReranker()


def _build_generator() -> IGenerator:
    if settings.llm_backend == "mlx":
        from app.infrastructure.mlx_generator import MLXGenerator
        return MLXGenerator()
    if settings.llm_backend == "ollama":
        from app.infrastructure.ollama_generator import OllamaGenerator
        return OllamaGenerator()
    from app.infrastructure.vllm_generator import VLLMGenerator
    return VLLMGenerator()


_embedder: IEmbedder = _build_embedder()
_reranker: IReranker = _build_reranker()
_generator: IGenerator = _build_generator()
_chunker = SemanticChunker(embedder=_embedder, threshold=settings.semantic_chunker_threshold)


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


async def get_generate_usecase(
    session: Annotated[AsyncSession, Depends(get_db)],
) -> GenerateUsecase:
    return GenerateUsecase(
        embedder=_embedder,
        vector_store=PgVectorStore(session),
        reranker=_reranker,
        generator=_generator,
    )


IngestDep = Annotated[IngestUsecase, Depends(get_ingest_usecase)]
ManageDep = Annotated[ManageUsecase, Depends(get_manage_usecase)]
SearchDep = Annotated[SearchUsecase, Depends(get_search_usecase)]
GenerateDep = Annotated[GenerateUsecase, Depends(get_generate_usecase)]
