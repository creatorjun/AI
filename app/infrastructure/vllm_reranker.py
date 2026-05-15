# app/infrastructure/vllm_reranker.py
import asyncio
import json

from openai import AsyncOpenAI

from app.config import settings
from app.domain.models import SearchResult
from app.domain.ports import IReranker

_RERANK_SYSTEM_PROMPT = (
    "You are a relevance scoring assistant. "
    "Given a query and a passage, output a single JSON object: {\"score\": <float 0.0-1.0>}. "
    "Score 1.0 means perfectly relevant, 0.0 means completely irrelevant. "
    "Output only the JSON object, no explanation."
)

_client = AsyncOpenAI(
    base_url=settings.vllm_base_url,
    api_key=settings.vllm_api_key,
)


class VLLMReranker(IReranker):
    def __init__(self) -> None:
        self._client = _client
        self._model = settings.vllm_model

    async def _score_single(self, query: str, content: str) -> float:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _RERANK_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Query: {query}\n\nPassage: {content}",
                },
            ],
            temperature=0.0,
            max_tokens=32,
        )
        raw = response.choices[0].message.content.strip()
        try:
            return float(json.loads(raw)["score"])
        except (json.JSONDecodeError, KeyError, ValueError):
            return 0.0

    async def rerank(
        self, query: str, results: list[SearchResult], top_k: int
    ) -> list[SearchResult]:
        scores = await asyncio.gather(
            *[self._score_single(query, r.content) for r in results]
        )
        ranked = sorted(
            [r.model_copy(update={"score": s}) for r, s in zip(results, scores)],
            key=lambda x: x.score,
            reverse=True,
        )
        return ranked[:top_k]


class NoOpReranker(IReranker):
    async def rerank(
        self, query: str, results: list[SearchResult], top_k: int
    ) -> list[SearchResult]:
        return results[:top_k]