# tests/unit/test_hybrid_rrf.py
from datetime import datetime, timezone

import pytest

from app.domain.models import SearchResult


def make_result(doc_id: str, chunk_index: int = 0, score: float = 0.9) -> SearchResult:
    return SearchResult(
        doc_id=doc_id,
        chunk_index=chunk_index,
        content=f"{doc_id} 에 대한 내용",
        score=score,
        trust_tier=3,
        tags=[],
        valid_from=datetime(2024, 1, 1, tzinfo=timezone.utc),
        valid_to=None,
        source_path="test.txt",
        recorded_at=datetime.now(tz=timezone.utc),
    )


def rrf_merge(
    vector_results: list[SearchResult],
    fulltext_results: list[SearchResult],
    top_k: int,
    alpha: float = 0.5,
    rrf_k: int = 60,
) -> list[SearchResult]:
    scores: dict[tuple[str, int], float] = {}
    doc_map: dict[tuple[str, int], SearchResult] = {}

    for rank, result in enumerate(vector_results):
        key = (result.doc_id, result.chunk_index)
        scores[key] = scores.get(key, 0.0) + alpha / (rrf_k + rank + 1)
        doc_map[key] = result

    for rank, result in enumerate(fulltext_results):
        key = (result.doc_id, result.chunk_index)
        scores[key] = scores.get(key, 0.0) + (1.0 - alpha) / (rrf_k + rank + 1)
        doc_map[key] = result

    sorted_keys = sorted(scores, key=lambda k: scores[k], reverse=True)
    return [
        doc_map[k].model_copy(update={"score": scores[k]})
        for k in sorted_keys[:top_k]
    ]


class TestRRFMerge:
    def test_alpha_05_weights_both_equally(self) -> None:
        v_results = [make_result("doc-v")]
        f_results = [make_result("doc-f")]
        merged = rrf_merge(v_results, f_results, top_k=2, alpha=0.5)
        scores = {r.doc_id: r.score for r in merged}
        assert scores["doc-v"] == pytest.approx(scores["doc-f"])

    def test_alpha_1_gives_all_weight_to_vector(self) -> None:
        v_results = [make_result("doc-v")]
        f_results = [make_result("doc-f")]
        merged = rrf_merge(v_results, f_results, top_k=2, alpha=1.0)
        scores = {r.doc_id: r.score for r in merged}
        assert scores["doc-v"] > 0
        assert scores["doc-f"] == pytest.approx(0.0)

    def test_alpha_0_gives_all_weight_to_fulltext(self) -> None:
        v_results = [make_result("doc-v")]
        f_results = [make_result("doc-f")]
        merged = rrf_merge(v_results, f_results, top_k=2, alpha=0.0)
        scores = {r.doc_id: r.score for r in merged}
        assert scores["doc-f"] > 0
        assert scores["doc-v"] == pytest.approx(0.0)

    def test_doc_appearing_in_both_gets_higher_score(self) -> None:
        shared = make_result("doc-shared")
        only_vector = make_result("doc-vector-only")
        merged = rrf_merge(
            vector_results=[shared, only_vector],
            fulltext_results=[shared],
            top_k=3,
            alpha=0.5,
        )
        scores = {r.doc_id: r.score for r in merged}
        assert scores["doc-shared"] > scores["doc-vector-only"]

    def test_top_k_limits_results(self) -> None:
        v_results = [make_result(f"doc-{i}") for i in range(10)]
        f_results = [make_result(f"doc-{i}") for i in range(10)]
        merged = rrf_merge(v_results, f_results, top_k=3, alpha=0.5)
        assert len(merged) == 3

    def test_results_are_sorted_by_score_descending(self) -> None:
        v_results = [make_result("doc-a"), make_result("doc-b"), make_result("doc-c")]
        f_results = [make_result("doc-c"), make_result("doc-b")]
        merged = rrf_merge(v_results, f_results, top_k=3, alpha=0.5)
        for i in range(len(merged) - 1):
            assert merged[i].score >= merged[i + 1].score

    def test_empty_vector_results(self) -> None:
        f_results = [make_result("doc-f")]
        merged = rrf_merge([], f_results, top_k=5, alpha=0.5)
        assert len(merged) == 1
        assert merged[0].doc_id == "doc-f"

    def test_empty_fulltext_results(self) -> None:
        v_results = [make_result("doc-v")]
        merged = rrf_merge(v_results, [], top_k=5, alpha=0.5)
        assert len(merged) == 1
        assert merged[0].doc_id == "doc-v"

    def test_both_empty_returns_empty(self) -> None:
        merged = rrf_merge([], [], top_k=5, alpha=0.5)
        assert merged == []
