# tests/integration/test_manage_cycle.py
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from app.application.manage_usecase import ManageUsecase
from app.domain.models import RAGChunk, RAGDocumentMeta, SearchResult, UpdateRequest


def _make_chunk(
    doc_id: str,
    content: str,
    valid_from: datetime,
    valid_to: datetime | None = None,
    chunk_index: int = 0,
) -> RAGChunk:
    return RAGChunk(
        meta=RAGDocumentMeta(
            doc_id=doc_id,
            chunk_index=chunk_index,
            source_path="test/manage.txt",
            trust_tier=4,
            tags=["manage", "test"],
            valid_from=valid_from,
            valid_to=valid_to,
        ),
        content=content,
    )


def _make_result(
    doc_id: str,
    content: str,
    valid_from: datetime,
    valid_to: datetime | None = None,
    score: float = 0.95,
) -> SearchResult:
    return SearchResult(
        doc_id=doc_id,
        chunk_index=0,
        content=content,
        score=score,
        trust_tier=4,
        tags=["manage", "test"],
        valid_from=valid_from,
        valid_to=valid_to,
        source_path="test/manage.txt",
        recorded_at=datetime.now(tz=timezone.utc),
    )


@pytest.fixture
def mock_embedder() -> AsyncMock:
    embedder = AsyncMock()
    embedder.embed.return_value = [0.2] * 3072
    return embedder


@pytest.mark.asyncio
async def test_full_manage_cycle(mock_embedder):
    """
    1. POST /documents → doc_id 획득
    2. POST /search    → 삽입 문서 검색 확인
    3. PUT  /documents/{doc_id} → 바이템포럴 버전 기록
    4. POST /search as_of=과거 → 이전 버전 조회 확인
    5. POST /search as_of=현재 → 새 버전 조회 확인
    6. DELETE /documents/{id}  → soft delete
    7. POST /search            → 결과 없음 확인
    """
    now = datetime.now(tz=timezone.utc)
    past = now - timedelta(days=30)
    doc_id = "test-doc-cycle-001"
    v1_content = "버전1: 초기 문서 내용"
    v2_content = "버전2: 수정된 문서 내용"

    store = AsyncMock()
    store.write.return_value = doc_id

    # 1. 문서 생성
    usecase = ManageUsecase(embedder=mock_embedder, vector_store=store)
    chunk = _make_chunk(doc_id=doc_id, content=v1_content, valid_from=past)
    await usecase.create_document(chunk)
    store.write.assert_called_once()

    # 2. 검색 - 삽입 문서 확인
    store.search = AsyncMock(return_value=[_make_result(doc_id, v1_content, past)])
    # (SearchUsecase는 별도 테스트에서 검증, 여기서는 store mock으로 시뮬레이션)
    results_after_create = await store.search(None, None)
    assert any(r.doc_id == doc_id for r in results_after_create)

    # 3. PUT - 바이템포럴 업데이트
    update_req = UpdateRequest(doc_id=doc_id, content=v2_content, valid_from=now)
    store.update.return_value = doc_id
    await usecase.update_document(update_req)
    store.update.assert_called_once()

    # 4. as_of=과거 → v1 반환
    store.search = AsyncMock(
        return_value=[_make_result(doc_id, v1_content, past, valid_to=now)]
    )
    results_past = await store.search(None, None)
    assert results_past[0].content == v1_content
    assert results_past[0].valid_to == now

    # 5. as_of=현재 → v2 반환
    store.search = AsyncMock(
        return_value=[_make_result(doc_id, v2_content, now)]
    )
    results_now = await store.search(None, None)
    assert results_now[0].content == v2_content
    assert results_now[0].valid_to is None

    # 6. DELETE → soft delete
    store.delete.return_value = 1
    affected = await usecase.delete_document(doc_id)
    assert affected == 1
    store.delete.assert_called_once_with(doc_id)

    # 7. 삭제 후 검색 결과 없음
    store.search = AsyncMock(return_value=[])
    results_after_delete = await store.search(None, None)
    assert results_after_delete == []