# tests/integration/test_search.py
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from app.application.search_usecase import SearchUsecase
from app.domain.models import SearchRequest, SearchResult


def make_result(doc_id: str = "doc-001", score: float = 0.9) -> SearchResult:
    return SearchResult(
        doc_id=doc_id,
        chunk_index=0,
        content="테스트 검색 결과 내용입니다.",
        score=score,
        trust_tier=3,
        tags=["test"],
        valid_from=datetime(2024, 1, 1, tzinfo=timezone.utc),
        valid_to=None,
        source_path="test/doc.txt",
        recorded_at=datetime.now(tz=timezone.utc),
    )


@pytest.fixture
def mock_embedder() -> AsyncMock:
    embedder = AsyncMock()
    embedder.embed.return_value = [0.1] * 3072
    return embedder


@pytest.fixture
def mock_vector_store() -> AsyncMock:
    store = AsyncMock()
    store.search.return_value = [make_result()]
    return store


@pytest.fixture
def mock_reranker() -> AsyncMock:
    reranker = AsyncMock()
    reranker.rerank.return_value = [make_result()]
    return reranker


@pytest.fixture
def search_usecase(mock_embedder, mock_vector_store, mock_reranker) -> SearchUsecase:
    return SearchUsecase(
        embedder=mock_embedder,
        vector_store=mock_vector_store,
        reranker=mock_reranker,
    )


@pytest.mark.asyncio
async def test_search_calls_embedder_and_store(
    search_usecase, mock_embedder, mock_vector_store
):
    request = SearchRequest(query="테스트 쿼리", rerank=False)
    results = await search_usecase.rag_search(request)

    mock_embedder.embed.assert_called_once_with("테스트 쿼리")
    mock_vector_store.search.assert_called_once()
    assert len(results) == 1


@pytest.mark.asyncio
async def test_search_with_rerank_calls_reranker(
    search_usecase, mock_reranker, monkeypatch
):
    import app.application.search_usecase as su_module
    monkeypatch.setattr(su_module.settings, "reranker_enabled", True)

    request = SearchRequest(query="테스트 쿼리", rerank=True)
    await search_usecase.rag_search(request)

    mock_reranker.rerank.assert_called_once()


@pytest.mark.asyncio
async def test_search_skips_reranker_when_disabled(
    search_usecase, mock_reranker, monkeypatch
):
    import app.application.search_usecase as su_module
    monkeypatch.setattr(su_module.settings, "reranker_enabled", False)

    request = SearchRequest(query="테스트 쿼리", rerank=True)
    await search_usecase.rag_search(request)

    mock_reranker.rerank.assert_not_called()


@pytest.mark.asyncio
async def test_search_returns_empty_when_no_results(
    mock_embedder, mock_reranker
):
    store = AsyncMock()
    store.search.return_value = []
    usecase = SearchUsecase(
        embedder=mock_embedder,
        vector_store=store,
        reranker=mock_reranker,
    )
    request = SearchRequest(query="없는 내용")
    results = await usecase.rag_search(request)
    assert results == []