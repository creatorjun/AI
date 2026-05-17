# alembic/versions/0002_change_embedding_dim_1024.py
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_index("ix_rag_documents_embedding", table_name="rag_documents")
    op.alter_column(
        "rag_documents",
        "embedding",
        type_=Vector(1024),
        existing_nullable=True,
        postgresql_using="NULL::vector(1024)",
    )
    op.create_index(
        "ix_rag_documents_embedding",
        "rag_documents",
        ["embedding"],
        postgresql_using="ivfflat",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


def downgrade() -> None:
    op.drop_index("ix_rag_documents_embedding", table_name="rag_documents")
    op.alter_column(
        "rag_documents",
        "embedding",
        type_=Vector(3072),
        existing_nullable=True,
        postgresql_using="NULL::vector(3072)",
    )
    op.create_index(
        "ix_rag_documents_embedding",
        "rag_documents",
        ["embedding"],
        postgresql_using="ivfflat",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
