# app/api/schemas/search_schema.py
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class SearchAPIRequest(BaseModel):
    query: str
    as_of: datetime | None = None
    trust_tier_min: int | None = Field(default=None, ge=1, le=5)
    tags: list[str] | None = None
    top_k: int = Field(default=10, ge=1, le=100)
    search_mode: Literal["hybrid", "vector", "fulltext"] = "hybrid"
    rerank: bool = True


class SearchResultItem(BaseModel):
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


class SearchAPIResponse(BaseModel):
    results: list[SearchResultItem]
    total: int