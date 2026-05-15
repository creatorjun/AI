# tests/unit/test_evaluation.py
from __future__ import annotations

import math
from unittest.mock import AsyncMock, MagicMock

import pytest

from evaluation.dataset import DEFAULT_DATASET, EvalSample, load_dataset
from evaluation.experiment_matrix import (
    ExperimentConfig,
    ExperimentResult,
    ExperimentRunner,
    build_configs,
)
from evaluation.ragas_evaluator import (
    EvalReport,
    RagasEvaluator,
    SampleScore,
    _cosine,
    _token_overlap,
)
from evaluation.report import _to_row, save_csv, save_json


class TestHelperFunctions:
    def test_cosine_identical_vectors(self):
        v = [1.0, 0.0, 0.0]
        assert _cosine(v, v) == pytest.approx(1.0)

    def test_cosine_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert _cosine(a, b) == pytest.approx(0.0)

    def test_cosine_zero_vector(self):
        assert _cosine([0.0, 0.0], [1.0, 0.0]) == 0.0

    def test_token_overlap_identical(self):
        assert _token_overlap("hello world", "hello world") == pytest.approx(1.0)

    def test_token_overlap_no_overlap(self):
        assert _token_overlap("abc def", "xyz uvw") == pytest.approx(0.0)

    def test_token_overlap_partial(self):
        score = _token_overlap("a b c", "b c d")
        assert 0.0 < score < 1.0


class TestEvalSampleAndDataset:
    def test_default_dataset_length(self):
        dataset = load_dataset()
        assert len(dataset) == len(DEFAULT_DATASET)

    def test_eval_sample_fields(self):
        sample = load_dataset()[0]
        assert isinstance(sample, EvalSample)
        assert sample.question
        assert sample.ground_truth_answer
        assert isinstance(sample.ground_truth_contexts, list)

    def test_load_dataset_from_json(self, tmp_path):
        import json
        data = [
            {
                "question": "Q1",
                "ground_truth_answer": "A1",
                "ground_truth_contexts": ["C1"],
            }
        ]
        p = tmp_path / "eval.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        dataset = load_dataset(str(p))
        assert len(dataset) == 1
        assert dataset[0].question == "Q1"


class TestEvalReport:
    def test_avg_score_empty(self):
        report = EvalReport()
        assert report.avg == pytest.approx(0.0)

    def test_avg_score_computed(self):
        scores = [
            SampleScore(question="q", faithfulness=0.8, context_recall=0.6, answer_relevancy=0.7),
            SampleScore(question="q", faithfulness=0.4, context_recall=0.2, answer_relevancy=0.3),
        ]
        report = EvalReport(scores=scores)
        assert report.faithfulness == pytest.approx(0.6)
        assert report.context_recall == pytest.approx(0.4)
        assert report.answer_relevancy == pytest.approx(0.5)
        assert report.avg == pytest.approx((0.6 + 0.4 + 0.5) / 3.0)


class TestRagasEvaluator:
    def _make_evaluator(self):
        embedder = MagicMock()
        embedder.embed = AsyncMock(return_value=[1.0, 0.0, 0.0])
        embedder.embed_batch = AsyncMock(return_value=[[1.0, 0.0, 0.0]])
        vector_store = MagicMock()
        vector_store.search = AsyncMock(return_value=[])
        return RagasEvaluator(embedder, vector_store), embedder, vector_store

    @pytest.mark.asyncio
    async def test_evaluate_empty_retrieved(self):
        evaluator, _, _ = self._make_evaluator()
        sample = EvalSample(
            question="Q?",
            ground_truth_answer="A",
            ground_truth_contexts=["C"],
        )
        report = await evaluator.evaluate([sample], {"top_k": 3})
        assert len(report.scores) == 1
        assert report.context_recall == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_evaluate_perfect_context_recall(self):
        from app.domain.models import SearchResult
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        evaluator, embedder, vector_store = self._make_evaluator()
        sample = EvalSample(
            question="FastAPI?",
            ground_truth_answer="FastAPI는 비동기를 지원합니다",
            ground_truth_contexts=["FastAPI는 비동기를 지원합니다"],
        )
        result = SearchResult(
            doc_id="d1",
            chunk_index=0,
            content="FastAPI는 비동기를 지원합니다",
            score=0.9,
            trust_tier=3,
            tags=[],
            valid_from=now,
            valid_to=None,
            source_path="/test",
            recorded_at=now,
        )
        vector_store.search = AsyncMock(return_value=[result])
        report = await evaluator.evaluate([sample], {"top_k": 3})
        assert report.context_recall == pytest.approx(1.0)


class TestBuildConfigs:
    def test_non_hybrid_uses_only_alpha_05(self):
        configs = build_configs(
            search_modes=["vector", "fulltext"],
            alphas=[0.3, 0.5, 0.7],
        )
        for cfg in configs:
            assert cfg.hybrid_alpha == 0.5

    def test_hybrid_has_all_alphas(self):
        configs = build_configs(
            search_modes=["hybrid"],
            alphas=[0.3, 0.5, 0.7],
            rerank_flags=[False],
            parent_flags=[False],
        )
        alphas = {cfg.hybrid_alpha for cfg in configs}
        assert alphas == {0.3, 0.5, 0.7}

    def test_label_auto_generated(self):
        cfg = ExperimentConfig(
            search_mode="hybrid",
            hybrid_alpha=0.5,
            rerank=True,
            use_parent_context=False,
        )
        assert "hybrid" in cfg.label
        assert "0.5" in cfg.label


class TestReport:
    def test_to_row_keys(self):
        from evaluation.report import FIELDS
        cfg = ExperimentConfig(
            search_mode="vector", hybrid_alpha=0.5, rerank=False, use_parent_context=False
        )
        result = ExperimentResult(config=cfg, report=EvalReport())
        row = _to_row(result)
        for field in FIELDS:
            assert field in row

    def test_save_csv(self, tmp_path):
        cfg = ExperimentConfig(
            search_mode="vector", hybrid_alpha=0.5, rerank=False, use_parent_context=False
        )
        result = ExperimentResult(config=cfg, report=EvalReport())
        out = str(tmp_path / "out.csv")
        save_csv([result], out)
        import csv
        rows = list(csv.DictReader(open(out, encoding="utf-8")))
        assert len(rows) == 1

    def test_save_json(self, tmp_path):
        import json
        cfg = ExperimentConfig(
            search_mode="fulltext", hybrid_alpha=0.5, rerank=True, use_parent_context=True
        )
        result = ExperimentResult(config=cfg, report=EvalReport())
        out = str(tmp_path / "out.json")
        save_json([result], out)
        data = json.loads(open(out, encoding="utf-8").read())
        assert len(data) == 1
        assert data[0]["search_mode"] == "fulltext"
