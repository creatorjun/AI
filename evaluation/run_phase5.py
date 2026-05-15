# evaluation/run_phase5.py
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.database import get_session
from app.infrastructure.openai_embedder import OpenAIEmbedder
from app.infrastructure.pg_vector_store import PgVectorStore
from evaluation.dataset import load_dataset
from evaluation.experiment_matrix import ExperimentRunner, build_configs
from evaluation.report import print_report, save_csv, save_json


async def main(dataset_path: str | None = None, output_dir: str = "evaluation/results") -> None:
    dataset = load_dataset(dataset_path)
    print(f"Loaded {len(dataset)} evaluation samples.")

    embedder = OpenAIEmbedder(
        api_key=settings.openai_api_key,
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
    )

    async with get_session() as session:
        vector_store = PgVectorStore(session)
        runner = ExperimentRunner(embedder, vector_store, dataset)
        configs = build_configs(top_k=5)
        print(f"Running {len(configs)} experiment configurations...")
        results = await runner.run_all(configs)

    best = ExperimentRunner.__new__(ExperimentRunner)
    best = runner.best_config(results)

    print_report(results, best)
    save_csv(results, f"{output_dir}/results.csv")
    save_json(results, f"{output_dir}/results.json")
    print(f"Results saved to {output_dir}/")


if __name__ == "__main__":
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else None
    asyncio.run(main(dataset_path))
