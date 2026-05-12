# app/application/manage_usecase.py
from __future__ import annotations

from app.domain.models import RAGChunk, RAGDocumentMeta, UpdateRequest
from app.domain.ports import IEmbedder, IVectorStore


class ManageUsecase:
    def __init__(self, embedder: IEmbedder, vector_store: IVectorStore) -> None:
        self._embedder = embedder
        self._vector_store = vector_store

    async def create_document(self, chunk: RAGChunk) -> str:
        embedding = await self._embedder.embed(chunk.content)
        chunk_with_embedding = chunk.model_copy(update={"embedding": embedding})
        return await self._vector_store.write(chunk_with_embedding)

    async def update_document(self, request: UpdateRequest) -> str:
        embedding = await self._embedder.embed(request.content)
        return await self._vector_store.update(request, embedding)

    async def delete_document(self, doc_id: str) -> int:
        return await self._vector_store.delete(doc_id)

    async def get_document(self, doc_id: str) -> list[RAGChunk]:
        return await self._vector_store.get_by_doc_id(doc_id)