# tests/unit/test_models.py
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.domain.models import RAGDocumentMeta, SearchRequest, SearchResult, UpdateRequest, WriteRequest


def test_trust_tier_range_valid():
    for tier in range(1, 6):
        meta = RAGDocumentMeta(
            doc_id="doc-001",
            chunk_index=0,
            source_path="test/file.txt",
            trust_tier=tier,
            valid_from=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        assert meta.trust_tier == tier


def test_trust_tier_out_of_range():
    with pytest.raises(ValidationError):
        RAGDocumentMeta(
            doc_id="doc-001",
            chunk_index=0,
            source_path="test/file.txt",
            trust_tier=6,
            valid_from=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )


def test_search_request_invalid_mode():
    with pytest.raises(ValidationError):
        SearchRequest(query="테스트", search_mode="invalid_mode")


def test_search_request_defaults():
    req = SearchRequest(query="테스트")
    assert req.search_mode == "hybrid"
    assert req.top_k == 10
    assert req.rerank is True
    assert req.hybrid_alpha == pytest.approx(0.5)
    assert req.use_parent_context is False


def test_write_request_auto_doc_id_is_none():
    req = WriteRequest(
        content="내용",
        source_path="a.txt",
        trust_tier=2,
        valid_from=datetime.now(timezone.utc),
    )
    assert req.doc_id is None


def test_search_request_hybrid_alpha_bounds():
    req_min = SearchRequest(query="q", hybrid_alpha=0.0)
    req_max = SearchRequest(query="q", hybrid_alpha=1.0)
    assert req_min.hybrid_alpha == 0.0
    assert req_max.hybrid_alpha == 1.0


def test_search_request_hybrid_alpha_out_of_bounds():
    with pytest.raises(ValidationError):
        SearchRequest(query="q", hybrid_alpha=1.1)
    with pytest.raises(ValidationError):
        SearchRequest(query="q", hybrid_alpha=-0.1)


def test_search_request_use_parent_context_default_false():
    req = SearchRequest(query="콘텐스")
    assert req.use_parent_context is False


def test_search_request_use_parent_context_true():
    req = SearchRequest(query="콘텐스", use_parent_context=True)
    assert req.use_parent_context is True


def test_search_result_parent_content_default_none():
    result = SearchResult(
        doc_id="doc-001",
        chunk_index=0,
        content="내용",
        score=0.9,
        trust_tier=3,
        tags=[],
        valid_from=datetime.now(timezone.utc),
        valid_to=None,
        source_path="a.txt",
        recorded_at=datetime.now(timezone.utc),
    )
    assert result.parent_content is None


def test_search_result_parent_content_populated():
    result = SearchResult(
        doc_id="doc-001",
        chunk_index=0,
        content="자식 청크",
        score=0.9,
        trust_tier=3,
        tags=[],
        valid_from=datetime.now(timezone.utc),
        valid_to=None,
        source_path="a.txt",
        recorded_at=datetime.now(timezone.utc),
        parent_content="부모 도큐먼트 전체 내용",
    )
    assert result.parent_content == "부모 도큐먼트 전체 내용"
