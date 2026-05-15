# evaluation/experiment_matrix.py
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Literal

from app.domain.ports import IEmbedder, IVectorStore
from evaluation.dataset import EvalSample
from evaluation.ragas_evaluator import EvalReport, RagasEvaluator


@dataclass
class ExperimentConfig:
    search_mode: Literal["hybrid", "vector", "fulltext"]
    hybrid_alpha: float
    rerank: bool
    use_parent_context: bool
    top_k: int = 5
    label: str = ""

    def __post_init__(self) -> None:
        if not self.label:
            rerank_str = "rerank" if self.rerank else "no-rerank"
            parent_str = "parent" if self.use_parent_context else "no-parent"
            self.label = (
                f"{self.search_mode}_alpha{self.hybrid_alpha:.1f}"
                f"_{rerank_str}_{parent_str}"
            )


@dataclass
class ExperimentResult:
    config: ExperimentConfig
    report: EvalReport
    error: str = ""

    @property
    def faithfulness(self) -> float:
        return self.report.faithfulness

    @property
    def context_recall(self) -> float:
        return self.report.context_recall

    @property
    def answer_relevancy(self) -> float:
        return self.report.answer_relevancy

    @property
    def avg_score(self) -> float:
        return self.report.avg


DEFAULT_SEARCH_MODES: list[Literal["hybrid", "vector", "fulltext"]] = [
    "hybrid",
    "vector",
    "fulltext",
]
DEFAULT_ALPHAS: list[float] = [0.3, 0.5, 0.7]
DEFAULT_RERANK_FLAGS: list[bool] = [True, False]
DEFAULT_PARENT_FLAGS: list[bool] = [False, True]


def build_configs(
    search_modes: list[str] | None = None,
    alphas: list[float] | None = None,
    rerank_flags: list[bool] | None = None,
    parent_flags: list[bool] | None = None,
    top_k: int = 5,
) -> list[ExperimentConfig]:
    modes = search_modes or DEFAULT_SEARCH_MODES
    _alphas = alphas or DEFAULT_ALPHAS
    _reranks = rerank_flags if rerank_flags is not None else DEFAULT_RERANK_FLAGS
    _parents = parent_flags if parent_flags is not None else DEFAULT_PARENT_FLAGS

    configs: list[ExperimentConfig] = []
    for mode, alpha, rerank, parent in product(modes, _alphas, _reranks, _parents):
        if mode != "hybrid" and alpha != 0.5:
            continue
        configs.append(
            ExperimentConfig(
                search_mode=mode,
                hybrid_alpha=alpha,
                rerank=rerank,
                use_parent_context=parent,
                top_k=top_k,
            )
        )
    return configs


class ExperimentRunner:
    def __init__(
        self,
        embedder: IEmbedder,
        vector_store: IVectorStore,
        dataset: list[EvalSample],
    ) -> None:
        self._evaluator = RagasEvaluator(embedder, vector_store)
        self._dataset = dataset

    async def run_all(
        self, configs: list[ExperimentConfig]
    ) -> list[ExperimentResult]:
        results: list[ExperimentResult] = []
        for cfg in configs:
            result = await self._run_one(cfg)
            results.append(result)
        return results

    async def _run_one(self, cfg: ExperimentConfig) -> ExperimentResult:
        try:
            kwargs = {
                "search_mode": cfg.search_mode,
                "hybrid_alpha": cfg.hybrid_alpha,
                "rerank": cfg.rerank,
                "use_parent_context": cfg.use_parent_context,
                "top_k": cfg.top_k,
            }
            report = await self._evaluator.evaluate(self._dataset, kwargs)
            return ExperimentResult(config=cfg, report=report)
        except Exception as exc:
            from evaluation.ragas_evaluator import EvalReport
            return ExperimentResult(config=cfg, report=EvalReport(), error=str(exc))

    def best_config(self, results: list[ExperimentResult]) -> ExperimentResult | None:
        valid = [r for r in results if not r.error]
        if not valid:
            return None
        return max(valid, key=lambda r: r.avg_score)
