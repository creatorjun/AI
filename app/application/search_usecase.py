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

        if request.use_parent_context and results:
            results = await self._attach_parent_context(results)

        return results

    async def _attach_parent_context(self, results: list[SearchResult]) -> list[SearchResult]:
        parent_ids = [
            r.doc_id
            for r in results
            if r.doc_id is not None
        ]
        unique_parent_ids = list(dict.fromkeys(parent_ids))
        parent_map = await self._vector_store.get_parent_chunks(unique_parent_ids)

        return [
            r.model_copy(update={"parent_content": parent_map.get(r.doc_id)})
            for r in results
        ]
