# tests/unit/test_chunker.py
import pytest

from app.infrastructure.chunker import FixedChunker, HierarchicalChunker, SentenceChunker


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