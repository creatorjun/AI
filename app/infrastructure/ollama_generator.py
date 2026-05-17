# app/infrastructure/ollama_generator.py
from __future__ import annotations

from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from app.config import settings
from app.domain.models import GenerateRequest, GenerateResponse, SearchResult
from app.domain.ports import IGenerator
from app.infrastructure.vllm_generator import _build_messages, _extract_sources


class OllamaGenerator(IGenerator):
    def __init__(self) -> None:
        self._client = AsyncOpenAI(
            base_url=f"{settings.ollama_base_url.rstrip('/')}/v1",
            api_key=settings.ollama_api_key,
        )
        self._model = settings.ollama_model

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
