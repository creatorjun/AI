# app/infrastructure/orm_models.py
import uuid

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    CheckConstraint,
    DateTime,
    Integer,
    SmallInteger,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class RAGDocumentORM(Base):
    __tablename__ = "rag_documents"
    __table_args__ = (
        UniqueConstraint("doc_id", "chunk_index"),
        CheckConstraint("trust_tier BETWEEN 1 AND 5", name="trust_tier_range"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    doc_id: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    parent_doc_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(3072), nullable=True)
    content_tsv: Mapped[str | None] = mapped_column(TSVECTOR, nullable=True)
    trust_tier: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    tags: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, server_default="{}"
    )
    source_path: Mapped[str] = mapped_column(Text, nullable=False)
    extra_meta: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default="{}")
    valid_from: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    valid_to: Mapped[DateTime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    recorded_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    deleted_at: Mapped[DateTime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )