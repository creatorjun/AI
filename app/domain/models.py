# app/domain/models.py
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class RAGDocumentMeta(BaseModel):
    doc_id: str
    chunk_index: int
    parent_doc_id: str | None = None
    source_path: str
    trust_tier: int = Field(..., ge=1, le=5)
    tags: list[str] = []
    extra_meta: dict = {}
    valid_from: datetime
    valid_to: datetime | None = None
    recorded_at: datetime | None = None


class RAGChunk(BaseModel):
    meta: RAGDocumentMeta
    content: str
    embedding: list[float] | None = None


class SearchRequest(BaseModel):
    query: str
    as_of: datetime | None = None
    trust_tier_min: int | None = None
    tags: list[str] | None = None
    top_k: int = 10
    search_mode: Literal["hybrid", "vector", "fulltext"] = "hybrid"
    rerank: bool = True
    hybrid_alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    use_parent_context: bool = False


class SearchResult(BaseModel):
    doc_id: str
    chunk_index: int
    content: str
    score: float
    trust_tier: int
    tags: list[str]
    valid_from: datetime
    valid_to: datetime | None
    source_path: str
    recorded_at: datetime
    parent_content: str | None = None


class UpdateRequest(BaseModel):
    doc_id: str
    content: str
    trust_tier: int | None = None
    tags: list[str] | None = None
    extra_meta: dict | None = None
    valid_from: datetime | None = None
    valid_to: datetime | None = None


class WriteRequest(BaseModel):
    content: str
    source_path: str
    trust_tier: int = Field(default=3, ge=1, le=5)
    tags: list[str] = []
    extra_meta: dict = {}
    valid_from: datetime
    valid_to: datetime | None = None
    doc_id: str | None = None
    parent_doc_id: str | None = None


class GenerateRequest(BaseModel):
    query: str
    contexts: list[SearchResult]
    system_prompt: str | None = None
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=64, le=8192)


class GenerateResponse(BaseModel):
    answer: str
    sources: list[str]
    model: str
