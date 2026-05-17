# app/infrastructure/mlx_generator.py
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from functools import partial

from app.config import settings
from app.domain.models import GenerateRequest, GenerateResponse
from app.domain.ports import IGenerator
from app.infrastructure.vllm_generator import _build_messages, _extract_sources


def _load_mlx_model(model_path: str):
    try:
        from mlx_lm import load
        return load(model_path)
    except ImportError as e:
        raise RuntimeError(
            "mlx-lm is required for MLX backend. Install with: pip install mlx-lm"
        ) from e


def _generate_sync(model, tokenizer, messages: list[dict], max_tokens: int, temperature: float) -> str:
    from mlx_lm import generate
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return generate(
        model, tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        temp=temperature,
        verbose=False,
    )


class MLXGenerator(IGenerator):
    def __init__(self) -> None:
        self._model, self._tokenizer = _load_mlx_model(settings.mlx_model)
        self._model_name = settings.mlx_model

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        messages = _build_messages(request)
        loop = asyncio.get_running_loop()
        fn = partial(
            _generate_sync,
            self._model, self._tokenizer,
            messages, request.max_tokens, request.temperature,
        )
        answer = await loop.run_in_executor(None, fn)
        return GenerateResponse(
            answer=answer.strip(),
            sources=_extract_sources(request.contexts),
            model=self._model_name,
        )

    async def generate_stream(self, request: GenerateRequest) -> AsyncIterator[str]:
        response = await self.generate(request)
        for word in response.answer.split(" "):
            yield word + " "
            await asyncio.sleep(0)
