# evaluation/report.py
from __future__ import annotations

import csv
import json
from pathlib import Path
from datetime import datetime

from evaluation.experiment_matrix import ExperimentResult
from evaluation.ragas_evaluator import EvalReport


FIELDS = [
    "label",
    "search_mode",
    "hybrid_alpha",
    "rerank",
    "use_parent_context",
    "top_k",
    "faithfulness",
    "context_recall",
    "answer_relevancy",
    "avg_score",
    "error",
]


def save_csv(results: list[ExperimentResult], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for r in results:
            writer.writerow(_to_row(r))


def save_json(results: list[ExperimentResult], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [_to_row(r) for r in results]
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def print_report(results: list[ExperimentResult], best: ExperimentResult | None = None) -> None:
    header = (
        f"{'Label':<50} "
        f"{'Faith':>6} {'Recall':>6} {'Relev':>6} {'Avg':>6} {'Err':<4}"
    )
    sep = "-" * len(header)
    print(f"\n{'=' * len(header)}")
    print(f"  Phase 5 RAGAS Evaluation Report  [{datetime.now().strftime('%Y-%m-%d %H:%M')}]")
    print(f"{'=' * len(header)}")
    print(header)
    print(sep)
    for r in sorted(results, key=lambda x: x.avg_score, reverse=True):
        err_mark = "ERR" if r.error else ""
        print(
            f"{r.config.label:<50} "
            f"{r.faithfulness:>6.3f} "
            f"{r.context_recall:>6.3f} "
            f"{r.answer_relevancy:>6.3f} "
            f"{r.avg_score:>6.3f} "
            f"{err_mark:<4}"
        )
    print(sep)
    if best:
        print(f"\n  Best Config  : {best.config.label}")
        print(f"  Faithfulness : {best.faithfulness:.4f}")
        print(f"  ContextRecall: {best.context_recall:.4f}")
        print(f"  AnswerRelev  : {best.answer_relevancy:.4f}")
        print(f"  Avg Score    : {best.avg_score:.4f}")
    print(f"{'=' * len(header)}\n")


def _to_row(r: ExperimentResult) -> dict:
    return {
        "label": r.config.label,
        "search_mode": r.config.search_mode,
        "hybrid_alpha": r.config.hybrid_alpha,
        "rerank": r.config.rerank,
        "use_parent_context": r.config.use_parent_context,
        "top_k": r.config.top_k,
        "faithfulness": round(r.faithfulness, 4),
        "context_recall": round(r.context_recall, 4),
        "answer_relevancy": round(r.answer_relevancy, 4),
        "avg_score": round(r.avg_score, 4),
        "error": r.error,
    }
