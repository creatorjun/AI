# evaluation/dataset.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class EvalSample(BaseModel):
    question: str
    ground_truth_answer: str
    ground_truth_contexts: list[str]


DEFAULT_DATASET: list[dict[str, Any]] = [
    {
        "question": "FastAPI에서 비동기 처리를 지원하는 방식은?",
        "ground_truth_answer": "FastAPI는 async/await 문법을 통해 비동기 처리를 완전히 지원하며 uvicorn과 함께 고성능 비동기 서버를 구성할 수 있습니다.",
        "ground_truth_contexts": [
            "FastAPI는 Python 기반의 고성능 웹 프레임워크입니다.",
            "비동기 처리를 완전히 지원하며 Pydantic을 통한 데이터 검증을 제공합니다.",
        ],
    },
    {
        "question": "pgvector를 사용하는 이유는 무엇인가?",
        "ground_truth_answer": "pgvector는 PostgreSQL 익스텐션으로, 외부 벡터DB 없이 PostgreSQL 안에서 직접 벡터 유사도 검색을 수행할 수 있게 해줍니다.",
        "ground_truth_contexts": [
            "pgvector 익스텐션을 통해 벡터 검색 기능을 PostgreSQL에서 직접 수행합니다.",
        ],
    },
    {
        "question": "클린 아키텍처에서 포트-어댑터 패턴의 역할은?",
        "ground_truth_answer": "포트-어댑터 패턴은 도메인 코드와 인프라 구현을 인터페이스로 분리하여 구체 구현체를 자유롭게 교체할 수 있도록 합니다.",
        "ground_truth_contexts": [
            "클린 아키텍처를 적용하면 도메인 로직과 인프라 코드를 명확히 분리할 수 있습니다.",
            "포트-어댑터 패턴은 인터페이스 교체를 용이하게 합니다.",
        ],
    },
    {
        "question": "RAG에서 hybrid 검색 모드란?",
        "ground_truth_answer": "hybrid 검색은 벡터 유사도 검색과 풀텍스트 키워드 검색을 RRF(Reciprocal Rank Fusion)로 결합하여 두 방식의 장점을 모두 활용하는 검색 전략입니다.",
        "ground_truth_contexts": [
            "hybrid 모드는 RRF(Reciprocal Rank Fusion)로 vector + fulltext를 상호 보완합니다.",
        ],
    },
    {
        "question": "바이템포럴 설계에서 valid_from과 recorded_at의 차이는?",
        "ground_truth_answer": "valid_from은 해당 정보가 현실에서 사실로 유효해진 시점(사실 시간)이고, recorded_at은 시스템에 기록된 시점(기록 시간)으로 두 시간 축을 독립적으로 관리합니다.",
        "ground_truth_contexts": [
            "valid_from / valid_to는 사실 시간(Fact Time)으로 해당 내용이 현실에서 유효한 기간을 나타냅니다.",
            "recorded_at은 기록 시간(Record Time)으로 시스템에 기록된 시점입니다.",
        ],
    },
]


def load_dataset(path: str | None = None) -> list[EvalSample]:
    if path is None:
        return [EvalSample(**item) for item in DEFAULT_DATASET]
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [EvalSample(**item) for item in raw]
