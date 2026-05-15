# evaluation/ragas_evaluator.py
from __future__ import annotations

import math
from dataclasses import dataclass, field

from app.domain.models import SearchRequest, SearchResult
from app.domain.ports import IEmbedder, IVectorStore
from evaluation.dataset import EvalSample


@dataclass
class SampleScore:
    question: str
    faithfulness: float
    context_recall: float
    answer_relevancy: float

    @property
    def avg(self) -> float:
        return (self.faithfulness + self.context_recall + self.answer_relevancy) / 3.0


@dataclass
class EvalReport:
    scores: list[SampleScore] = field(default_factory=list)

    @property
    def faithfulness(self) -> float:
        if not self.scores:
            return 0.0
        return sum(s.faithfulness for s in self.scores) / len(self.scores)

    @property
    def context_recall(self) -> float:
        if not self.scores:
            return 0.0
        return sum(s.context_recall for s in self.scores) / len(self.scores)

    @property
    def answer_relevancy(self) -> float:
        if not self.scores:
            return 0.0
        return sum(s.answer_relevancy for s in self.scores) / len(self.scores)

    @property
    def avg(self) -> float:
        return (self.faithfulness + self.context_recall + self.answer_relevancy) / 3.0


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _token_overlap(a: str, b: str) -> float:
    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    return len(intersection) / len(tokens_a | tokens_b)


class RagasEvaluator:
    def __init__(self, embedder: IEmbedder, vector_store: IVectorStore) -> None:
        self._embedder = embedder
        self._vector_store = vector_store

    async def evaluate(
        self,
        dataset: list[EvalSample],
        search_request_kwargs: dict,
    ) -> EvalReport:
        report = EvalReport()
        for sample in dataset:
            retrieved = await self._retrieve(sample.question, search_request_kwargs)
            score = await self._score_sample(sample, retrieved)
            report.scores.append(score)
        return report

    async def _retrieve(self, question: str, kwargs: dict) -> list[SearchResult]:
        embedding = await self._embedder.embed(question)
        req = SearchRequest(query=question, **kwargs)
        return await self._vector_store.search(req, embedding)

    async def _score_sample(
        self,
        sample: EvalSample,
        retrieved: list[SearchResult],
    ) -> SampleScore:
        retrieved_texts = [r.content for r in retrieved]

        faithfulness = self._calc_faithfulness(
            sample.ground_truth_answer, retrieved_texts
        )
        context_recall = self._calc_context_recall(
            sample.ground_truth_contexts, retrieved_texts
        )
        answer_relevancy = await self._calc_answer_relevancy(
            sample.question, sample.ground_truth_answer
        )

        return SampleScore(
            question=sample.question,
            faithfulness=faithfulness,
            context_recall=context_recall,
            answer_relevancy=answer_relevancy,
        )

    def _calc_faithfulness(self, answer: str, contexts: list[str]) -> float:
        if not contexts:
            return 0.0
        scores = [_token_overlap(answer, ctx) for ctx in contexts]
        return max(scores)

    def _calc_context_recall(
        self, ground_truth_contexts: list[str], retrieved_texts: list[str]
    ) -> float:
        if not ground_truth_contexts:
            return 1.0
        if not retrieved_texts:
            return 0.0
        hit = 0
        for gt_ctx in ground_truth_contexts:
            best = max(_token_overlap(gt_ctx, rt) for rt in retrieved_texts)
            if best >= 0.15:
                hit += 1
        return hit / len(ground_truth_contexts)

    async def _calc_answer_relevancy(self, question: str, answer: str) -> float:
        q_emb = await self._embedder.embed(question)
        a_emb = await self._embedder.embed(answer)
        return max(0.0, _cosine(q_emb, a_emb))
