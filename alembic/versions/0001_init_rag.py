# alembic/versions/0001_init_rag.py
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_bigm")

    op.create_table(
        "rag_documents",
        sa.Column("id", sa.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("doc_id", sa.Text, nullable=False),
        sa.Column("chunk_index", sa.Integer, nullable=False),
        sa.Column("parent_doc_id", sa.Text, nullable=True),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("embedding", Vector(3072), nullable=True),
        sa.Column("content_tsv", TSVECTOR, nullable=True),
        sa.Column("trust_tier", sa.SmallInteger, nullable=False),
        sa.Column("tags", sa.ARRAY(sa.Text), nullable=False, server_default="{}"),
        sa.Column("source_path", sa.Text, nullable=False),
        sa.Column("extra_meta", JSONB, nullable=False, server_default="{}"),
        sa.Column("valid_from", sa.DateTime(timezone=True), nullable=False),
        sa.Column("valid_to", sa.DateTime(timezone=True), nullable=True),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("doc_id", "chunk_index"),
        sa.CheckConstraint("trust_tier BETWEEN 1 AND 5", name="trust_tier_range"),
    )

    op.create_index(
        "ix_rag_documents_embedding",
        "rag_documents",
        ["embedding"],
        postgresql_using="ivfflat",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
    op.create_index(
        "ix_rag_documents_content_tsv",
        "rag_documents",
        ["content_tsv"],
        postgresql_using="gin",
    )
    op.create_index(
        "ix_rag_documents_tags",
        "rag_documents",
        ["tags"],
        postgresql_using="gin",
    )
    op.create_index(
        "ix_rag_documents_trust_tier_valid",
        "rag_documents",
        ["trust_tier", "valid_from", "valid_to"],
    )


def downgrade() -> None:
    op.drop_table("rag_documents")
    op.execute("DROP EXTENSION IF EXISTS pg_bigm")
    op.execute("DROP EXTENSION IF EXISTS vector")