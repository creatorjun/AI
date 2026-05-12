# tests/unit/test_models.py
import pytest
from datetime import datetime, timezone
from pydantic import ValidationError

from app.domain.models import RAGDocumentMeta, WriteRequest, SearchRequest


def test_trust_tier_range_valid():
    meta = RAGDocumentMeta(
        doc_id="doc1", chunk_index=0, source_path="a.txt",
        trust_tier=3, valid_from=datetime.now(timezone.utc)
    )
    assert meta.trust_tier == 3


def test_trust_tier_out_of_range():
    with pytest.raises(ValidationError):
        RAGDocumentMeta(
            doc_id="doc1", chunk_index=0, source_path="a.txt",
            trust_tier=6, valid_from=datetime.now(timezone.utc)
        )


def test_search_request_invalid_mode():
    with pytest.raises(ValidationError):
        SearchRequest(query="test", search_mode="invalid")


def test_search_request_defaults():
    req = SearchRequest(query="test")
    assert req.top_k == 10
    assert req.search_mode == "hybrid"
    assert req.as_of is None


def test_write_request_auto_doc_id_is_none():
    req = WriteRequest(
        content="내용", source_path="a.txt", trust_tier=2,
        valid_from=datetime.now(timezone.utc)
    )
    assert req.doc_id is None