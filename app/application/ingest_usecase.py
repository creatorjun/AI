# app/application/ingest_usecase.py
from __future__ import annotations

import asyncio
import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path

from app.domain.models import RAGChunk, RAGDocumentMeta
from app.domain.ports import IChunker, IEmbedder, IVectorStore

_SUPPORTED_EXTENSIONS = {".txt", ".md", ".rst", ".csv", ".json"}


def _make_doc_id(source_path: str) -> str:
    return hashlib.sha256(source_path.encode()).hexdigest()[:16]


class IngestUsecase:
    def __init__(
        self,
        embedder: IEmbedder,
        chunker: IChunker,
        vector_store: IVectorStore,
    ) -> None:
        self._embedder = embedder
        self._chunker = chunker
        self._vector_store = vector_store

    async def ingest_file(
        self,
        file_path: str,
        trust_tier: int = 3,
        tags: list[str] | None = None,
        valid_from: datetime | None = None,
        valid_to: datetime | None = None,
    ) -> str:
        path = Path(file_path)
        if path.suffix not in _SUPPORTED_EXTENSIONS:
            raise ValueError(f"지원하지 않는 파일 형식: {path.suffix}")

        content = path.read_text(encoding="utf-8")
        doc_id = _make_doc_id(file_path)
        vf = valid_from or datetime.now(tz=timezone.utc)

        text_chunks = self._chunker.chunk(content, file_path)
        embeddings = await self._embedder.embed_batch(text_chunks)

        chunks = [
            RAGChunk(
                meta=RAGDocumentMeta(
                    doc_id=doc_id,
                    chunk_index=idx,
                    source_path=file_path,
                    trust_tier=trust_tier,
                    tags=tags or [],
                    valid_from=vf,
                    valid_to=valid_to,
                ),
                content=text,
                embedding=embedding,
            )
            for idx, (text, embedding) in enumerate(zip(text_chunks, embeddings))
        ]

        await self._vector_store.write_batch(chunks)
        return doc_id

    async def ingest_folder(
        self,
        folder_path: str,
        trust_tier: int = 3,
        tags: list[str] | None = None,
        valid_from: datetime | None = None,
        valid_to: datetime | None = None,
    ) -> tuple[int, str]:
        folder = Path(folder_path)
        files = [
            f for f in folder.rglob("*")
            if f.is_file() and f.suffix in _SUPPORTED_EXTENSIONS
        ]

        async def _ingest(f: Path) -> None:
            await self.ingest_file(
                str(f),
                trust_tier=trust_tier,
                tags=tags,
                valid_from=valid_from,
                valid_to=valid_to,
            )

        await asyncio.gather(*[_ingest(f) for f in files])
        return len(files), folder_path

    async def sync_folder(self, folder_path: str) -> dict[str, int]:
        folder = Path(folder_path)
        disk_paths = {
            str(f)
            for f in folder.rglob("*")
            if f.is_file() and f.suffix in _SUPPORTED_EXTENSIONS
        }
        stored_paths = set(await self._vector_store.get_source_paths())

        to_add = disk_paths - stored_paths
        to_delete = stored_paths - disk_paths
        to_update = disk_paths & stored_paths

        added = 0
        for path in to_add:
            await self.ingest_file(path)
            added += 1

        deleted = 0
        for path in to_delete:
            doc_id = _make_doc_id(path)
            await self._vector_store.delete(doc_id)
            deleted += 1

        updated = 0
        for path in to_update:
            doc_id = _make_doc_id(path)
            stored_mtime = None
            chunks = await self._vector_store.get_by_doc_id(doc_id)
            if chunks:
                stored_mtime = chunks[0].meta.recorded_at
            disk_mtime = datetime.fromtimestamp(
                os.path.getmtime(path), tz=timezone.utc
            )
            if stored_mtime is None or disk_mtime > stored_mtime:
                await self.ingest_file(path)
                updated += 1

        return {"added": added, "updated": updated, "deleted": deleted} 