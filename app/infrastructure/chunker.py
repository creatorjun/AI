# app/infrastructure/chunker.py
from __future__ import annotations

import re

from app.domain.ports import IChunker


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
        raw = re.split(r"(?<=[.!?。？！])\s+", content.strip())
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