# tests/integration/test_ingest.py
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.application.ingest_usecase import IngestUsecase
from app.infrastructure.chunker import SentenceChunker


@pytest.fixture
def mock_embedder() -> AsyncMock:
    embedder = AsyncMock()
    embedder.embed_batch.return_value = [[0.1] * 3072]
    return embedder


@pytest.fixture
def mock_vector_store() -> AsyncMock:
    store = AsyncMock()
    store.write_batch.return_value = ["doc-abc123"]
    store.get_source_paths.return_value = []
    return store


@pytest.fixture
def ingest_usecase(mock_embedder, mock_vector_store) -> IngestUsecase:
    return IngestUsecase(
        embedder=mock_embedder,
        chunker=SentenceChunker(),
        vector_store=mock_vector_store,
    )


@pytest.mark.asyncio
async def test_ingest_file_calls_embedder(
    tmp_path, ingest_usecase, mock_embedder, mock_vector_store
):
    f = tmp_path / "test.txt"
    f.write_text("테스트 문서입니다. 내용이 있습니다.", encoding="utf-8")

    doc_id = await ingest_usecase.ingest_file(
        file_path=str(f),
        trust_tier=3,
        tags=["test"],
        valid_from=datetime.now(tz=timezone.utc),
    )

    assert doc_id is not None
    mock_embedder.embed_batch.assert_called_once()
    mock_vector_store.write_batch.assert_called_once()


@pytest.mark.asyncio
async def test_ingest_file_unsupported_extension(tmp_path, ingest_usecase):
    f = tmp_path / "test.pdf"
    f.write_text("content")

    with pytest.raises(ValueError, match="지원하지 않는 파일 형식"):
        await ingest_usecase.ingest_file(str(f))


@pytest.mark.asyncio
async def test_ingest_folder_processes_all_files(
    tmp_path, ingest_usecase, mock_embedder, mock_vector_store
):
    mock_embedder.embed_batch.return_value = [[0.1] * 3072]

    for name in ["a.txt", "b.md", "c.rst"]:
        (tmp_path / name).write_text("내용입니다.", encoding="utf-8")
    (tmp_path / "ignore.pdf").write_text("무시")

    count, path = await ingest_usecase.ingest_folder(
        folder_path=str(tmp_path),
        valid_from=datetime.now(tz=timezone.utc),
    )

    assert count == 3
    assert mock_embedder.embed_batch.call_count == 3