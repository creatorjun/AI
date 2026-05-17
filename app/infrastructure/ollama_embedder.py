# app/infrastructure/ollama_embedder.py
import httpx

from app.config import settings
from app.domain.ports import IEmbedder


class OllamaEmbedder(IEmbedder):
    def __init__(self) -> None:
        self._base_url = settings.ollama_base_url.rstrip("/v1").rstrip("/")
        self._model = settings.ollama_embedding_model
        self._dimensions = settings.embedding_dimensions

    async def embed(self, text: str) -> list[float]:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self._base_url}/api/embed",
                json={"model": self._model, "input": text},
            )
            response.raise_for_status()
            return response.json()["embeddings"][0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self._base_url}/api/embed",
                json={"model": self._model, "input": texts},
            )
            response.raise_for_status()
            return response.json()["embeddings"]
