# app/api/schemas/manage_schema.py
from datetime import datetime

from pydantic import BaseModel, Field


class CreateDocumentRequest(BaseModel):
    content: str
    source_path: str
    trust_tier: int = Field(default=3, ge=1, le=5)
    tags: list[str] = []
    extra_meta: dict = {}
    valid_from: datetime
    valid_to: datetime | None = None
    doc_id: str | None = None
    parent_doc_id: str | None = None


class CreateDocumentResponse(BaseModel):
    doc_id: str


class UpdateDocumentRequest(BaseModel):
    content: str
    trust_tier: int | None = Field(default=None, ge=1, le=5)
    tags: list[str] | None = None
    extra_meta: dict | None = None
    valid_from: datetime | None = None
    valid_to: datetime | None = None


class UpdateDocumentResponse(BaseModel):
    doc_id: str


class DeleteDocumentResponse(BaseModel):
    doc_id: str
    affected_chunks: int


class ChunkItem(BaseModel):
    doc_id: str
    chunk_index: int
    content: str
    trust_tier: int
    tags: list[str]
    valid_from: datetime
    valid_to: datetime | None
    source_path: str
    recorded_at: datetime | None


class GetDocumentResponse(BaseModel):
    doc_id: str
    chunks: list[ChunkItem]