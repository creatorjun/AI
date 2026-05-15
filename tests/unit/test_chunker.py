# tests/unit/test_chunker.py
from unittest.mock import AsyncMock

import pytest

from app.infrastructure.chunker import FixedChunker, HierarchicalChunker, SemanticChunker, SentenceChunker


@pytest.fixture
def source_path() -> str:
    return "test/sample.txt"


class TestFixedChunker:
    def test_single_chunk_when_content_is_short(self, source_path: str) -> None:
        chunker = FixedChunker(chunk_size=512, overlap=64)
        chunks = chunker.chunk("short content", source_path)
        assert len(chunks) == 1
        assert chunks[0] == "short content"

    def test_multiple_chunks_with_overlap(self, source_path: str) -> None:
        tokens = ["word"] * 600
        content = " ".join(tokens)
        chunker = FixedChunker(chunk_size=512, overlap=64)
        chunks = chunker.chunk(content, source_path)
        assert len(chunks) >= 2

    def test_no_empty_chunks(self, source_path: str) -> None:
        chunker = FixedChunker(chunk_size=10, overlap=2)
        chunks = chunker.chunk("   ", source_path)
        assert chunks == []

    def test_chunk_size_respected(self, source_path: str) -> None:
        tokens = ["word"] * 1000
        content = " ".join(tokens)
        chunker = FixedChunker(chunk_size=100, overlap=0)
        chunks = chunker.chunk(content, source_path)
        for chunk in chunks[:-1]:
            assert len(chunk.split()) == 100


class TestSentenceChunker:
    def test_splits_by_sentence(self, source_path: str) -> None:
        content = "첫 번째 문장입니다. 두 번째 문장입니다. 세 번째 문장입니다."
        chunker = SentenceChunker(max_sentences=2, overlap_sentences=0)
        chunks = chunker.chunk(content, source_path)
        assert len(chunks) >= 1

    def test_overlap_sentences(self, source_path: str) -> None:
        sentences = [f"문장 {i}입니다." for i in range(10)]
        content = " ".join(sentences)
        chunker = SentenceChunker(max_sentences=3, overlap_sentences=1)
        chunks = chunker.chunk(content, source_path)
        assert len(chunks) > 1

    def test_no_empty_chunks(self, source_path: str) -> None:
        chunker = SentenceChunker()
        chunks = chunker.chunk("", source_path)
        assert chunks == []


class TestHierarchicalChunker:
    def test_chunk_returns_child_chunks(self, source_path: str) -> None:
        tokens = ["word"] * 2000
        content = " ".join(tokens)
        chunker = HierarchicalChunker(parent_chunk_size=1024, child_chunk_size=256, overlap=32)
        children = chunker.chunk(content, source_path)
        assert len(children) > 0

    def test_chunk_with_parents_returns_both(self, source_path: str) -> None:
        tokens = ["word"] * 2000
        content = " ".join(tokens)
        chunker = HierarchicalChunker(parent_chunk_size=1024, child_chunk_size=256, overlap=32)
        parents, children = chunker.chunk_with_parents(content, source_path)
        assert len(parents) > 0
        assert len(children) >= len(parents)

    def test_children_smaller_than_parents(self, source_path: str) -> None:
        tokens = ["word"] * 2000
        content = " ".join(tokens)
        chunker = HierarchicalChunker(parent_chunk_size=1024, child_chunk_size=256, overlap=0)
        parents, children = chunker.chunk_with_parents(content, source_path)
        avg_parent = sum(len(p.split()) for p in parents) / len(parents)
        avg_child = sum(len(c.split()) for c in children) / len(children)
        assert avg_child < avg_parent


class TestSemanticChunker:
    def _make_embedder(self, vectors: list[list[float]]) -> AsyncMock:
        embedder = AsyncMock()
        embedder.embed_batch.return_value = vectors
        return embedder

    @pytest.mark.asyncio
    async def test_single_sentence_returns_one_chunk(self, source_path: str) -> None:
        embedder = self._make_embedder([[1.0, 0.0]])
        chunker = SemanticChunker(embedder=embedder, threshold=0.85)
        result = await chunker.async_chunk("단일 문장입니다.", source_path)
        assert len(result) == 1
        assert result[0] == "단일 문장입니다."

    @pytest.mark.asyncio
    async def test_empty_content_returns_empty_list(self, source_path: str) -> None:
        embedder = self._make_embedder([])
        chunker = SemanticChunker(embedder=embedder, threshold=0.85)
        result = await chunker.async_chunk("", source_path)
        assert result == []

    @pytest.mark.asyncio
    async def test_high_similarity_merges_sentences(self, source_path: str) -> None:
        v = [1.0, 0.0]
        embedder = self._make_embedder([v, v, v])
        chunker = SemanticChunker(embedder=embedder, threshold=0.85)
        content = "첫 문장. 두 문장. 세 문장."
        result = await chunker.async_chunk(content, source_path)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_low_similarity_splits_sentences(self, source_path: str) -> None:
        v1 = [1.0, 0.0]
        v2 = [0.0, 1.0]
        embedder = self._make_embedder([v1, v2, v1])
        chunker = SemanticChunker(embedder=embedder, threshold=0.85)
        content = "첫 문장. 두 문장. 세 문장."
        result = await chunker.async_chunk(content, source_path)
        assert len(result) >= 2

    @pytest.mark.asyncio
    async def test_max_chunk_sentences_enforced(self, source_path: str) -> None:
        v = [1.0, 0.0]
        n = 10
        embedder = self._make_embedder([v] * n)
        chunker = SemanticChunker(embedder=embedder, threshold=0.99, max_chunk_sentences=3)
        sentences = ["문장." for _ in range(n)]
        content = " ".join(sentences)
        result = await chunker.async_chunk(content, source_path)
        for chunk in result[:-1]:
            assert len(chunk.split(".")) - 1 <= 3

    def test_cosine_same_vector_is_one(self) -> None:
        v = [1.0, 0.0, 0.0]
        assert SemanticChunker._cosine(v, v) == pytest.approx(1.0)

    def test_cosine_orthogonal_vectors_is_zero(self) -> None:
        v1 = [1.0, 0.0]
        v2 = [0.0, 1.0]
        assert SemanticChunker._cosine(v1, v2) == pytest.approx(0.0)

    def test_cosine_zero_vector_returns_zero(self) -> None:
        v1 = [0.0, 0.0]
        v2 = [1.0, 0.0]
        assert SemanticChunker._cosine(v1, v2) == 0.0
