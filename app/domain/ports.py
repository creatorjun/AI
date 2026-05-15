# app/domain/ports.py
from abc import ABC, abstractmethod

from app.domain.models import RAGChunk, SearchRequest, SearchResult, UpdateRequest


class IEmbedder(ABC):
    @abstractmethod
    async def embed(self, text: str) -> list[float]: ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


class IChunker(ABC):
    @abstractmethod
    def chunk(self, content: str, source_path: str) -> list[str]: ...


class IReranker(ABC):
    @abstractmethod
    async def rerank(
        self, query: str, results: list[SearchResult], top_k: int
    ) -> list[SearchResult]: ...


class IVectorStore(ABC):
    @abstractmethod
    async def write(self, chunk: RAGChunk) -> str: ...

    @abstractmethod
    async def write_batch(self, chunks: list[RAGChunk]) -> list[str]: ...

    @abstractmethod
    async def update(self, request: UpdateRequest, embedding: list[float]) -> str: ...

    @abstractmethod
    async def delete(self, doc_id: str) -> int: ...

    @abstractmethod
    async def search(
        self, request: SearchRequest, query_embedding: list[float]
    ) -> list[SearchResult]: ...

    @abstractmethod
    async def get_by_doc_id(self, doc_id: str) -> list[RAGChunk]: ...

    @abstractmethod
    async def get_source_paths(self) -> list[str]: ...

    @abstractmethod
    async def get_parent_chunks(self, parent_doc_ids: list[str]) -> dict[str, str]: ...
