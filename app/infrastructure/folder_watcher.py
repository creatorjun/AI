# app/infrastructure/folder_watcher.py
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from watchdog.events import FileCreatedEvent, FileDeletedEvent, FileModifiedEvent, FileSystemEventHandler
from watchdog.observers import Observer

from app.application.ingest_usecase import IngestUsecase, _make_doc_id, _SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)


class _RAGEventHandler(FileSystemEventHandler):
    def __init__(self, loop: asyncio.AbstractEventLoop, usecase: IngestUsecase) -> None:
        super().__init__()
        self._loop = loop
        self._usecase = usecase

    def _submit(self, coro) -> None:
        asyncio.run_coroutine_threadsafe(coro, self._loop)

    def on_created(self, event: FileCreatedEvent) -> None:
        if not event.is_directory and Path(event.src_path).suffix in _SUPPORTED_EXTENSIONS:
            logger.info("[FolderWatcher] created: %s", event.src_path)
            self._submit(self._usecase.ingest_file(event.src_path))

    def on_modified(self, event: FileModifiedEvent) -> None:
        if not event.is_directory and Path(event.src_path).suffix in _SUPPORTED_EXTENSIONS:
            logger.info("[FolderWatcher] modified: %s", event.src_path)
            self._submit(self._usecase.ingest_file(event.src_path))

    def on_deleted(self, event: FileDeletedEvent) -> None:
        if not event.is_directory and Path(event.src_path).suffix in _SUPPORTED_EXTENSIONS:
            logger.info("[FolderWatcher] deleted: %s", event.src_path)
            doc_id = _make_doc_id(event.src_path)
            self._submit(self._usecase._vector_store.delete(doc_id))


class FolderWatcher:
    def __init__(self, folder_path: str, usecase: IngestUsecase) -> None:
        self._folder_path = folder_path
        self._usecase = usecase
        self._observer: Observer | None = None

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        if not self._folder_path:
            return
        handler = _RAGEventHandler(loop, self._usecase)
        self._observer = Observer()
        self._observer.schedule(handler, self._folder_path, recursive=True)
        self._observer.start()
        logger.info("[FolderWatcher] watching: %s", self._folder_path)

    def stop(self) -> None:
        if self._observer and self._observer.is_alive():
            self._observer.stop()
            self._observer.join()
            logger.info("[FolderWatcher] stopped")
