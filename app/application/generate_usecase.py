# app/application/generate_usecase.py
from __future__ import annotations

from collections.abc import AsyncIterator

from app.config import settings
from app.domain.models import GenerateRequest, GenerateResponse, SearchRequest, SearchResult
from app.domain.ports import IEmbedder, IGenerator, IReranker, IVectorStore


class GenerateUsecase:
    def __init__(
        self,
        embedder: IEmbedder,
        vector_store: IVectorStore,
        reranker: IReranker,
        generator: IGenerator,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._reranker = reranker
        self._generator = generator

    async def _retrieve(self, request: SearchRequest) -> list[SearchResult]:
        query_embedding = await self._embedder.embed(request.query)
        results = await self._vector_store.search(request, query_embedding)
        if request.rerank and settings.reranker_enabled and results:
            results = await self._reranker.rerank(request.query, results, request.top_k)
        if request.use_parent_context and results:
            parent_ids = list(dict.fromkeys(r.doc_id for r in results if r.doc_id))
            parent_map = await self._vector_store.get_parent_chunks(parent_ids)
            results = [
                r.model_copy(update={"parent_content": parent_map.get(r.doc_id)})
                for r in results
            ]
        return results

    async def rag_generate(
        self,
        search_request: SearchRequest,
        system_prompt: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> GenerateResponse:
        contexts = await self._retrieve(search_request)
        gen_request = GenerateRequest(
            query=search_request.query,
            contexts=contexts,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return await self._generator.generate(gen_request)

    async def rag_generate_stream(
        self,
        search_request: SearchRequest,
        system_prompt: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> AsyncIterator[str]:
        contexts = await self._retrieve(search_request)
        gen_request = GenerateRequest(
            query=search_request.query,
            contexts=contexts,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        async for token in self._generator.generate_stream(gen_request):
            yield token
