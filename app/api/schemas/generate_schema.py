# app/api/schemas/generate_schema.py
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class GenerateAPIRequest(BaseModel):
    query: str
    as_of: datetime | None = None
    trust_tier_min: int | None = Field(default=None, ge=1, le=5)
    tags: list[str] | None = None
    top_k: int = Field(default=5, ge=1, le=20)
    search_mode: Literal["hybrid", "vector", "fulltext"] = "hybrid"
    rerank: bool = True
    hybrid_alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    use_parent_context: bool = True
    system_prompt: str | None = None
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=64, le=8192)
    stream: bool = False


class GenerateAPIResponse(BaseModel):
    answer: str
    sources: list[str]
    model: str
