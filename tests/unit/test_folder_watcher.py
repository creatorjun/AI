# tests/unit/test_folder_watcher.py
import asyncio
from unittest.mock import MagicMock, patch

import pytest

from app.infrastructure.folder_watcher import FolderWatcher, _RAGEventHandler


class TestFolderWatcher:
    def test_start_does_nothing_when_folder_path_empty(self) -> None:
        usecase = MagicMock()
        watcher = FolderWatcher(folder_path="", usecase=usecase)
        loop = asyncio.new_event_loop()
        try:
            watcher.start(loop)
            assert watcher._observer is None
        finally:
            loop.close()

    def test_start_creates_observer_when_folder_set(self, tmp_path) -> None:
        usecase = MagicMock()
        watcher = FolderWatcher(folder_path=str(tmp_path), usecase=usecase)
        loop = asyncio.new_event_loop()
        try:
            with patch("app.infrastructure.folder_watcher.Observer") as mock_obs_cls:
                mock_obs = MagicMock()
                mock_obs_cls.return_value = mock_obs
                watcher.start(loop)
                mock_obs.schedule.assert_called_once()
                mock_obs.start.assert_called_once()
        finally:
            loop.close()
            watcher.stop()

    def test_stop_does_nothing_when_not_started(self) -> None:
        usecase = MagicMock()
        watcher = FolderWatcher(folder_path="", usecase=usecase)
        watcher.stop()


class TestRAGEventHandler:
    def _make_handler(self):
        loop = asyncio.new_event_loop()
        usecase = MagicMock()
        usecase.ingest_file = MagicMock(return_value="doc-id")

        vector_store = MagicMock()
        vector_store.delete = MagicMock(return_value=1)
        usecase._vector_store = vector_store

        handler = _RAGEventHandler(loop=loop, usecase=usecase)
        return handler, usecase, loop

    def test_on_created_supported_file_submits_ingest(self, tmp_path) -> None:
        handler, usecase, loop = self._make_handler()
        event = MagicMock()
        event.is_directory = False
        event.src_path = str(tmp_path / "doc.txt")
        with patch("asyncio.run_coroutine_threadsafe") as mock_submit:
            handler.on_created(event)
            mock_submit.assert_called_once()
        loop.close()

    def test_on_modified_supported_file_submits_ingest(self, tmp_path) -> None:
        handler, usecase, loop = self._make_handler()
        event = MagicMock()
        event.is_directory = False
        event.src_path = str(tmp_path / "doc.md")
        with patch("asyncio.run_coroutine_threadsafe") as mock_submit:
            handler.on_modified(event)
            mock_submit.assert_called_once()
        loop.close()

    def test_on_deleted_supported_file_submits_delete(self, tmp_path) -> None:
        handler, usecase, loop = self._make_handler()
        event = MagicMock()
        event.is_directory = False
        event.src_path = str(tmp_path / "doc.txt")
        with patch("app.infrastructure.folder_watcher._make_doc_id", return_value="doc-id"), \
             patch("asyncio.run_coroutine_threadsafe") as mock_submit:
            handler.on_deleted(event)
            mock_submit.assert_called_once()
        loop.close()

    def test_on_created_unsupported_file_is_ignored(self, tmp_path) -> None:
        handler, usecase, loop = self._make_handler()
        event = MagicMock()
        event.is_directory = False
        event.src_path = str(tmp_path / "image.png")
        with patch("asyncio.run_coroutine_threadsafe") as mock_submit:
            handler.on_created(event)
            mock_submit.assert_not_called()
        loop.close()

    def test_on_created_directory_is_ignored(self, tmp_path) -> None:
        handler, usecase, loop = self._make_handler()
        event = MagicMock()
        event.is_directory = True
        event.src_path = str(tmp_path / "subdir")
        with patch("asyncio.run_coroutine_threadsafe") as mock_submit:
            handler.on_created(event)
            mock_submit.assert_not_called()
        loop.close()
