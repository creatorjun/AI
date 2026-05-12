# tests/unit/test_bitemporal.py
from datetime import datetime, timedelta, timezone

import pytest

from app.domain.models import RAGChunk, RAGDocumentMeta


def make_chunk(
    doc_id: str = "doc-001",
    chunk_index: int = 0,
    valid_from: datetime | None = None,
    valid_to: datetime | None = None,
) -> RAGChunk:
    vf = valid_from or datetime(2024, 1, 1, tzinfo=timezone.utc)
    return RAGChunk(
        meta=RAGDocumentMeta(
            doc_id=doc_id,
            chunk_index=chunk_index,
            source_path="test/doc.txt",
            trust_tier=3,
            valid_from=vf,
            valid_to=valid_to,
        ),
        content=f"청크 내용 {chunk_index}",
    )


def is_valid_at(chunk: RAGChunk, as_of: datetime) -> bool:
    meta = chunk.meta
    if meta.valid_from > as_of:
        return False
    if meta.valid_to is not None and meta.valid_to <= as_of:
        return False
    return True


class TestBitemporalLogic:
    def test_chunk_valid_when_no_valid_to(self) -> None:
        chunk = make_chunk()
        now = datetime.now(tz=timezone.utc)
        assert is_valid_at(chunk, now) is True

    def test_chunk_invalid_before_valid_from(self) -> None:
        future = datetime.now(tz=timezone.utc) + timedelta(days=30)
        chunk = make_chunk(valid_from=future)
        now = datetime.now(tz=timezone.utc)
        assert is_valid_at(chunk, now) is False

    def test_chunk_invalid_after_valid_to(self) -> None:
        past_from = datetime(2023, 1, 1, tzinfo=timezone.utc)
        past_to = datetime(2023, 6, 1, tzinfo=timezone.utc)
        chunk = make_chunk(valid_from=past_from, valid_to=past_to)
        now = datetime.now(tz=timezone.utc)
        assert is_valid_at(chunk, now) is False

    def test_chunk_valid_within_range(self) -> None:
        past = datetime(2023, 1, 1, tzinfo=timezone.utc)
        future = datetime(2099, 1, 1, tzinfo=timezone.utc)
        chunk = make_chunk(valid_from=past, valid_to=future)
        now = datetime.now(tz=timezone.utc)
        assert is_valid_at(chunk, now) is True

    def test_as_of_query_returns_historical_version(self) -> None:
        v1_from = datetime(2023, 1, 1, tzinfo=timezone.utc)
        v1_to = datetime(2024, 1, 1, tzinfo=timezone.utc)
        v2_from = datetime(2024, 1, 1, tzinfo=timezone.utc)

        v1 = make_chunk(doc_id="doc-001", chunk_index=0, valid_from=v1_from, valid_to=v1_to)
        v2 = make_chunk(doc_id="doc-001", chunk_index=1, valid_from=v2_from)

        as_of_past = datetime(2023, 6, 1, tzinfo=timezone.utc)
        as_of_now = datetime.now(tz=timezone.utc)

        assert is_valid_at(v1, as_of_past) is True
        assert is_valid_at(v2, as_of_past) is False
        assert is_valid_at(v1, as_of_now) is False
        assert is_valid_at(v2, as_of_now) is True

    def test_multiple_versions_only_one_active(self) -> None:
        versions = [
            make_chunk(
                chunk_index=i,
                valid_from=datetime(2023 + i, 1, 1, tzinfo=timezone.utc),
                valid_to=datetime(2024 + i, 1, 1, tzinfo=timezone.utc) if i < 2 else None,
            )
            for i in range(3)
        ]
        now = datetime.now(tz=timezone.utc)
        active = [c for c in versions if is_valid_at(c, now)]
        assert len(active) == 1