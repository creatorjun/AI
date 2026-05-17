# app/infrastructure/vllm_generator.py
from __future__ import annotations

from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from app.config import settings
from app.domain.models import GenerateRequest, GenerateResponse, SearchResult
from app.domain.ports import IGenerator

_DEFAULT_SYSTEM = (
    "You are a helpful assistant. Answer the user's question using ONLY the provided context. "
    "If the context does not contain enough information, say so clearly. "
    "Cite sources by their source_path when relevant."
)


def _build_messages(request: GenerateRequest) -> list[dict]:
    system = request.system_prompt or _DEFAULT_SYSTEM
    context_block = "\n\n".join(
        f"[{i+1}] (source: {r.source_path})\n{r.parent_content or r.content}"
        for i, r in enumerate(request.contexts)
    )
    user_content = f"Context:\n{context_block}\n\nQuestion: {request.query}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


def _extract_sources(contexts: list[SearchResult]) -> list[str]:
    seen: set[str] = set()
    sources: list[str] = []
    for r in contexts:
        if r.source_path not in seen:
            seen.add(r.source_path)
            sources.append(r.source_path)
    return sources


class VLLMGenerator(IGenerator):
    def __init__(self) -> None:
        self._client = AsyncOpenAI(
            base_url=settings.vllm_base_url,
            api_key=settings.vllm_api_key,
        )
        self._model = settings.vllm_model

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=_build_messages(request),
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        return GenerateResponse(
            answer=response.choices[0].message.content.strip(),
            sources=_extract_sources(request.contexts),
            model=self._model,
        )

    async def generate_stream(self, request: GenerateRequest) -> AsyncIterator[str]:
        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=_build_messages(request),
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
