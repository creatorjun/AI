# app/infrastructure/pg_vector_store.py
from __future__ import annotations

from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import and_, func, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.models import RAGChunk, RAGDocumentMeta, SearchRequest, SearchResult, UpdateRequest
from app.domain.ports import IVectorStore
from app.infrastructure.orm_models import RAGDocumentORM


class PgVectorStore(IVectorStore):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    def _orm_to_chunk(self, row: RAGDocumentORM) -> RAGChunk:
        meta = RAGDocumentMeta(
            doc_id=row.doc_id,
            chunk_index=row.chunk_index,
            parent_doc_id=row.parent_doc_id,
            source_path=row.source_path,
            trust_tier=row.trust_tier,
            tags=row.tags or [],
            extra_meta=row.extra_meta or {},
            valid_from=row.valid_from,
            valid_to=row.valid_to,
            recorded_at=row.recorded_at,
        )
        return RAGChunk(
            meta=meta,
            content=row.content,
            embedding=list(row.embedding) if row.embedding is not None else None,
        )

    def _orm_to_result(self, row: RAGDocumentORM, score: float) -> SearchResult:
        return SearchResult(
            doc_id=row.doc_id,
            chunk_index=row.chunk_index,
            content=row.content,
            score=score,
            trust_tier=row.trust_tier,
            tags=row.tags or [],
            valid_from=row.valid_from,
            valid_to=row.valid_to,
            source_path=row.source_path,
            recorded_at=row.recorded_at,
        )

    def _build_orm(self, chunk: RAGChunk) -> RAGDocumentORM:
        return RAGDocumentORM(
            doc_id=chunk.meta.doc_id,
            chunk_index=chunk.meta.chunk_index,
            parent_doc_id=chunk.meta.parent_doc_id,
            content=chunk.content,
            embedding=chunk.embedding,
            trust_tier=chunk.meta.trust_tier,
            tags=chunk.meta.tags,
            source_path=chunk.meta.source_path,
            extra_meta=chunk.meta.extra_meta,
            valid_from=chunk.meta.valid_from,
            valid_to=chunk.meta.valid_to,
        )

    async def write(self, chunk: RAGChunk) -> str:
        orm = self._build_orm(chunk)
        self._session.add(orm)
        await self._session.flush()
        return chunk.meta.doc_id

    async def write_batch(self, chunks: list[RAGChunk]) -> list[str]:
        orm_list = [self._build_orm(chunk) for chunk in chunks]
        self._session.add_all(orm_list)
        await self._session.flush()
        await self._session.commit()
        return [chunk.meta.doc_id for chunk in chunks]

    async def update(self, request: UpdateRequest, embedding: list[float]) -> str:
        now = datetime.now(tz=timezone.utc)
        await self._session.execute(
            update(RAGDocumentORM)
            .where(
                and_(
                    RAGDocumentORM.doc_id == request.doc_id,
                    RAGDocumentORM.deleted_at.is_(None),
                    RAGDocumentORM.valid_to.is_(None),
                )
            )
            .values(valid_to=now)
        )

        stmt = select(RAGDocumentORM).where(
            and_(
                RAGDocumentORM.doc_id == request.doc_id,
                RAGDocumentORM.deleted_at.is_(None),
            )
        ).order_by(RAGDocumentORM.chunk_index)
        result = await self._session.execute(stmt)
        existing = result.scalars().first()

        new_chunk = RAGChunk(
            meta=RAGDocumentMeta(
                doc_id=request.doc_id,
                chunk_index=0,
                parent_doc_id=existing.parent_doc_id if existing else None,
                source_path=existing.source_path if existing else "",
                trust_tier=request.trust_tier or (existing.trust_tier if existing else 3),
                tags=request.tags or (existing.tags if existing else []),
                extra_meta=request.extra_meta or (existing.extra_meta if existing else {}),
                valid_from=request.valid_from or now,
                valid_to=request.valid_to,
            ),
            content=request.content,
            embedding=embedding,
        )
        await self.write(new_chunk)
        await self._session.commit()
        return request.doc_id

    async def delete(self, doc_id: str) -> int:
        now = datetime.now(tz=timezone.utc)
        result = await self._session.execute(
            update(RAGDocumentORM)
            .where(
                and_(
                    RAGDocumentORM.doc_id == doc_id,
                    RAGDocumentORM.deleted_at.is_(None),
                )
            )
            .values(deleted_at=now)
        )
        await self._session.commit()
        return result.rowcount

    async def search(
        self, request: SearchRequest, query_embedding: list[float]
    ) -> list[SearchResult]:
        as_of = request.as_of or datetime.now(tz=timezone.utc)

        base_conditions = [
            RAGDocumentORM.deleted_at.is_(None),
            RAGDocumentORM.valid_from <= as_of,
            (RAGDocumentORM.valid_to.is_(None)) | (RAGDocumentORM.valid_to > as_of),
        ]
        if request.trust_tier_min is not None:
            base_conditions.append(RAGDocumentORM.trust_tier >= request.trust_tier_min)
        if request.tags:
            base_conditions.append(RAGDocumentORM.tags.overlap(request.tags))

        limit = request.top_k * 3

        if request.search_mode == "vector":
            return await self._vector_search(query_embedding, base_conditions, request.top_k)
        elif request.search_mode == "fulltext":
            return await self._fulltext_search(request.query, base_conditions, request.top_k)
        else:
            return await self._hybrid_search(query_embedding, request.query, base_conditions, request.top_k, limit)

    async def _vector_search(
        self, embedding: list[float], conditions: list, top_k: int
    ) -> list[SearchResult]:
        distance_expr = RAGDocumentORM.embedding.cosine_distance(embedding).label("distance")

        stmt = (
            select(RAGDocumentORM, distance_expr)
            .where(and_(*conditions))
            .order_by(distance_expr)
            .limit(top_k)
        )
        rows = await self._session.execute(stmt)
        return [
            self._orm_to_result(row, 1.0 - float(dist))
            for row, dist in rows.all()
        ]

    async def _fulltext_search(
        self, query: str, conditions: list, top_k: int
    ) -> list[SearchResult]:
        tsquery = func.plainto_tsquery("simple", query)
        rank_expr = func.ts_rank(RAGDocumentORM.content_tsv, tsquery).label("rank")

        stmt = (
            select(RAGDocumentORM, rank_expr)
            .where(and_(*conditions, RAGDocumentORM.content_tsv.op("@@")(tsquery)))
            .order_by(rank_expr.desc())
            .limit(top_k)
        )
        rows = await self._session.execute(stmt)
        return [self._orm_to_result(row, float(rank)) for row, rank in rows.all()]

    async def _hybrid_search(
        self,
        embedding: list[float],
        query: str,
        conditions: list,
        top_k: int,
        limit: int,
    ) -> list[SearchResult]:
        vector_results = await self._vector_search(embedding, conditions, limit)
        fulltext_results = await self._fulltext_search(query, conditions, limit)

        rrf_k = 60
        scores: dict[tuple[str, int], float] = {}
        doc_map: dict[tuple[str, int], SearchResult] = {}

        for rank, result in enumerate(vector_results):
            key = (result.doc_id, result.chunk_index)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
            doc_map[key] = result

        for rank, result in enumerate(fulltext_results):
            key = (result.doc_id, result.chunk_index)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
            doc_map[key] = result

        sorted_keys = sorted(scores, key=lambda k: scores[k], reverse=True)
        return [
            doc_map[k].model_copy(update={"score": scores[k]})
            for k in sorted_keys[:top_k]
        ]

    async def get_by_doc_id(self, doc_id: str) -> list[RAGChunk]:
        stmt = select(RAGDocumentORM).where(
            and_(
                RAGDocumentORM.doc_id == doc_id,
                RAGDocumentORM.deleted_at.is_(None),
            )
        ).order_by(RAGDocumentORM.chunk_index)
        result = await self._session.execute(stmt)
        return [self._orm_to_chunk(row) for row in result.scalars().all()]

    async def get_source_paths(self) -> list[str]:
        stmt = select(RAGDocumentORM.source_path).where(
            RAGDocumentORM.deleted_at.is_(None)
        ).distinct()
        result = await self._session.execute(stmt)
        return [row for row in result.scalars().all()]