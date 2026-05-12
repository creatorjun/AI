# tests/unit/test_models.py
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.domain.models import RAGDocumentMeta, UpdateRequest, WriteRequest, SearchRequest


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


def test_write_request_auto_doc_id_is_none():
    req = WriteRequest(
        content="내용",
        source_path="a.txt",
        trust_tier=2,
        valid_from=datetime.now(timezone.utc),
    )
    assert req.doc_id is None