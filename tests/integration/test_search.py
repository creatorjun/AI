# tests/integration/test_search.py
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from app.application.search_usecase import SearchUsecase
from app.domain.models import SearchRequest, SearchResult


def make_result(
    doc_id: str = "doc-001",
    score: float = 0.9,
    parent_content: str | None = None,
) -> SearchResult:
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
        parent_content=parent_content,
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
    store.get_parent_chunks.return_value = {"doc-001": "부모 도큐먼트 전체 내용"}
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


@pytest.mark.asyncio
async def test_search_passes_hybrid_alpha_to_store(
    search_usecase, mock_vector_store
):
    request = SearchRequest(query="쿼리", rerank=False, hybrid_alpha=0.3)
    await search_usecase.rag_search(request)
    call_args = mock_vector_store.search.call_args
    passed_request: SearchRequest = call_args[0][0]
    assert passed_request.hybrid_alpha == pytest.approx(0.3)


@pytest.mark.asyncio
async def test_use_parent_context_attaches_parent_content(
    search_usecase, mock_vector_store
):
    mock_vector_store.search.return_value = [make_result(doc_id="doc-001")]
    mock_vector_store.get_parent_chunks.return_value = {"doc-001": "부모 전체 컨텐츠"}

    request = SearchRequest(query="쿼리", rerank=False, use_parent_context=True)
    results = await search_usecase.rag_search(request)

    assert results[0].parent_content == "부모 전체 컨텐츠"
    mock_vector_store.get_parent_chunks.assert_called_once_with(["doc-001"])


@pytest.mark.asyncio
async def test_use_parent_context_false_skips_parent_fetch(
    search_usecase, mock_vector_store
):
    request = SearchRequest(query="쿼리", rerank=False, use_parent_context=False)
    await search_usecase.rag_search(request)
    mock_vector_store.get_parent_chunks.assert_not_called()


@pytest.mark.asyncio
async def test_use_parent_context_deduplicates_doc_ids(
    mock_embedder, mock_reranker
):
    store = AsyncMock()
    store.search.return_value = [
        make_result(doc_id="doc-001"),
        make_result(doc_id="doc-001"),
        make_result(doc_id="doc-002"),
    ]
    store.get_parent_chunks.return_value = {
        "doc-001": "부모-001",
        "doc-002": "부모-002",
    }
    usecase = SearchUsecase(
        embedder=mock_embedder,
        vector_store=store,
        reranker=mock_reranker,
    )
    request = SearchRequest(query="쿼리", rerank=False, use_parent_context=True)
    results = await usecase.rag_search(request)

    called_ids = store.get_parent_chunks.call_args[0][0]
    assert called_ids.count("doc-001") == 1
    assert results[0].parent_content == "부모-001"
    assert results[2].parent_content == "부모-002"


@pytest.mark.asyncio
async def test_use_parent_context_with_empty_results(
    mock_embedder, mock_reranker
):
    store = AsyncMock()
    store.search.return_value = []
    usecase = SearchUsecase(
        embedder=mock_embedder,
        vector_store=store,
        reranker=mock_reranker,
    )
    request = SearchRequest(query="쿼리", rerank=False, use_parent_context=True)
    results = await usecase.rag_search(request)
    assert results == []
    store.get_parent_chunks.assert_not_called()
