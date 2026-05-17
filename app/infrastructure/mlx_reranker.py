# app/infrastructure/mlx_reranker.py
import asyncio
import json
from functools import partial

from app.config import settings
from app.domain.models import SearchResult
from app.domain.ports import IReranker

_RERANK_SYSTEM_PROMPT = (
    "You are a relevance scoring assistant. "
    "Given a query and a passage, output a single JSON object: {\"score\": <float 0.0-1.0>}. "
    "Score 1.0 means perfectly relevant, 0.0 means completely irrelevant. "
    "Output only the JSON object, no explanation."
)


def _load_mlx_model(model_path: str):
    try:
        from mlx_lm import load
        return load(model_path)
    except ImportError as e:
        raise RuntimeError(
            "mlx-lm package is required for MLX backend. "
            "Install with: pip install mlx-lm"
        ) from e


def _generate_score_sync(model, tokenizer, query: str, content: str, max_tokens: int) -> float:
    from mlx_lm import generate

    messages = [
        {"role": "system", "content": _RERANK_SYSTEM_PROMPT},
        {"role": "user", "content": f"Query: {query}\n\nPassage: {content}"},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    raw = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
    try:
        return float(json.loads(raw.strip())["score"])
    except (json.JSONDecodeError, KeyError, ValueError):
        return 0.0


class MLXReranker(IReranker):
    def __init__(self) -> None:
        self._model, self._tokenizer = _load_mlx_model(settings.mlx_model)
        self._max_tokens = settings.mlx_max_tokens

    def _score_single_sync(self, query: str, content: str) -> float:
        return _generate_score_sync(
            self._model, self._tokenizer, query, content, self._max_tokens
        )

    async def _score_single(self, query: str, content: str) -> float:
        loop = asyncio.get_running_loop()
        fn = partial(self._score_single_sync, query, content)
        return await loop.run_in_executor(None, fn)

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
