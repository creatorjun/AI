# app/infrastructure/chunker.py
from __future__ import annotations

import math
import re

from app.domain.ports import IChunker, IEmbedder


class FixedChunker(IChunker):
    def __init__(self, chunk_size: int = 512, overlap: int = 64) -> None:
        self._chunk_size = chunk_size
        self._overlap = overlap

    def chunk(self, content: str, source_path: str) -> list[str]:
        tokens = content.split()
        chunks: list[str] = []
        start = 0
        while start < len(tokens):
            end = start + self._chunk_size
            chunks.append(" ".join(tokens[start:end]))
            start += self._chunk_size - self._overlap
        return [c for c in chunks if c.strip()]


class SentenceChunker(IChunker):
    def __init__(self, max_sentences: int = 5, overlap_sentences: int = 1) -> None:
        self._max_sentences = max_sentences
        self._overlap_sentences = overlap_sentences

    def _split_sentences(self, content: str) -> list[str]:
        raw = re.split(r"(?<=[.!?\u3002\uff1f\uff01])\s+", content.strip())
        return [s.strip() for s in raw if s.strip()]

    def chunk(self, content: str, source_path: str) -> list[str]:
        sentences = self._split_sentences(content)
        chunks: list[str] = []
        start = 0
        while start < len(sentences):
            end = start + self._max_sentences
            chunks.append(" ".join(sentences[start:end]))
            start += self._max_sentences - self._overlap_sentences
        return [c for c in chunks if c.strip()]


class HierarchicalChunker(IChunker):
    def __init__(
        self,
        parent_chunk_size: int = 1024,
        child_chunk_size: int = 256,
        overlap: int = 32,
    ) -> None:
        self._parent_chunker = FixedChunker(chunk_size=parent_chunk_size, overlap=0)
        self._child_chunker = FixedChunker(chunk_size=child_chunk_size, overlap=overlap)

    def chunk(self, content: str, source_path: str) -> list[str]:
        return self._child_chunker.chunk(content, source_path)

    def chunk_with_parents(self, content: str, source_path: str) -> tuple[list[str], list[str]]:
        parents = self._parent_chunker.chunk(content, source_path)
        children: list[str] = []
        for parent in parents:
            children.extend(self._child_chunker.chunk(parent, source_path))
        return parents, children


class SemanticChunker(IChunker):
    def __init__(
        self,
        embedder: IEmbedder,
        threshold: float = 0.85,
        max_chunk_sentences: int = 20,
    ) -> None:
        self._embedder = embedder
        self._threshold = threshold
        self._max_chunk_sentences = max_chunk_sentences
        self._splitter = SentenceChunker(max_sentences=1, overlap_sentences=0)

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    def chunk(self, content: str, source_path: str) -> list[str]:
        raise RuntimeError(
            "SemanticChunker는 async 컨텍스트에서 async_chunk()를 직접 호출해야 합니다."
        )

    async def async_chunk(self, content: str, source_path: str) -> list[str]:
        sentences = self._splitter._split_sentences(content)
        if not sentences:
            return []
        if len(sentences) == 1:
            return sentences

        embeddings = await self._embedder.embed_batch(sentences)

        chunks: list[str] = []
        current: list[str] = [sentences[0]]

        for i in range(1, len(sentences)):
            sim = self._cosine(embeddings[i - 1], embeddings[i])
            if sim >= self._threshold and len(current) < self._max_chunk_sentences:
                current.append(sentences[i])
            else:
                chunks.append(" ".join(current))
                current = [sentences[i]]

        if current:
            chunks.append(" ".join(current))

        return [c for c in chunks if c.strip()]