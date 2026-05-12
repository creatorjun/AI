# app/application/search_usecase.py
from __future__ import annotations

from app.config import settings
from app.domain.models import SearchRequest, SearchResult
from app.domain.ports import IEmbedder, IReranker, IVectorStore


class SearchUsecase:
    def __init__(
        self,
        embedder: IEmbedder,
        vector_store: IVectorStore,
        reranker: IReranker,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._reranker = reranker

    async def rag_search(self, request: SearchRequest) -> list[SearchResult]:
        query_embedding = await self._embedder.embed(request.query)
        results = await self._vector_store.search(request, query_embedding)

        if request.rerank and settings.reranker_enabled and results:
            results = await self._reranker.rerank(request.query, results, request.top_k)

        return results