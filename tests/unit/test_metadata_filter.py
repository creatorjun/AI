# tests/unit/test_metadata_filter.py
from datetime import datetime, timezone

import pytest

from app.domain.models import RAGDocumentMeta, SearchRequest


def make_meta(**kwargs) -> RAGDocumentMeta:
    defaults = dict(
        doc_id="doc-001",
        chunk_index=0,
        source_path="test/file.txt",
        trust_tier=3,
        tags=["python", "fastapi"],
        valid_from=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    defaults.update(kwargs)
    return RAGDocumentMeta(**defaults)


class TestTrustTierFilter:
    def test_trust_tier_in_valid_range(self) -> None:
        for tier in range(1, 6):
            meta = make_meta(trust_tier=tier)
            assert 1 <= meta.trust_tier <= 5

    def test_trust_tier_below_range_raises(self) -> None:
        with pytest.raises(ValueError):
            make_meta(trust_tier=0)

    def test_trust_tier_above_range_raises(self) -> None:
        with pytest.raises(ValueError):
            make_meta(trust_tier=6)


class TestTagsFilter:
    def test_tags_default_empty(self) -> None:
        meta = RAGDocumentMeta(
            doc_id="doc-002",
            chunk_index=0,
            source_path="test/file.txt",
            trust_tier=3,
            valid_from=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        assert meta.tags == []

    def test_tags_overlap_logic(self) -> None:
        meta = make_meta(tags=["python", "rag", "vector"])
        request_tags = ["rag", "llm"]
        overlap = set(meta.tags) & set(request_tags)
        assert len(overlap) > 0

    def test_no_tag_overlap(self) -> None:
        meta = make_meta(tags=["python"])
        request_tags = ["java", "go"]
        overlap = set(meta.tags) & set(request_tags)
        assert len(overlap) == 0


class TestSearchRequestDefaults:
    def test_default_search_mode_is_hybrid(self) -> None:
        req = SearchRequest(query="테스트 쿼리")
        assert req.search_mode == "hybrid"

    def test_default_top_k_is_10(self) -> None:
        req = SearchRequest(query="테스트 쿼리")
        assert req.top_k == 10

    def test_rerank_default_true(self) -> None:
        req = SearchRequest(query="테스트 쿼리")
        assert req.rerank is True

    def test_hybrid_alpha_default(self) -> None:
        req = SearchRequest(query="테스트 쿼리")
        assert req.hybrid_alpha == pytest.approx(0.5)

    def test_use_parent_context_default_false(self) -> None:
        req = SearchRequest(query="테스트 쿼리")
        assert req.use_parent_context is False
