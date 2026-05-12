# app/api/schemas/ingest_schema.py
from datetime import datetime

from pydantic import BaseModel, Field


class IngestFolderRequest(BaseModel):
    folder_path: str
    trust_tier: int = Field(default=3, ge=1, le=5)
    tags: list[str] = []
    valid_from: datetime
    valid_to: datetime | None = None


class IngestFolderResponse(BaseModel):
    queued_files: int
    folder_path: str


class IngestFileRequest(BaseModel):
    file_path: str
    trust_tier: int = Field(default=3, ge=1, le=5)
    tags: list[str] = []
    valid_from: datetime
    valid_to: datetime | None = None


class IngestFileResponse(BaseModel):
    doc_id: str
    source_path: str


class SyncFolderRequest(BaseModel):
    folder_path: str


class SyncFolderResponse(BaseModel):
    added: int
    updated: int
    deleted: int