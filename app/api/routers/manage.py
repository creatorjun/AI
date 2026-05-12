# app/api/routers/manage.py
import uuid

from fastapi import APIRouter, HTTPException

from app.api.deps import ManageDep
from app.api.schemas.manage_schema import (
    ChunkItem,
    CreateDocumentRequest,
    CreateDocumentResponse,
    DeleteDocumentResponse,
    GetDocumentResponse,
    UpdateDocumentRequest,
    UpdateDocumentResponse,
)
from app.domain.models import RAGChunk, RAGDocumentMeta, UpdateRequest

router = APIRouter(prefix="/documents", tags=["manage"])


@router.post("", response_model=CreateDocumentResponse)
async def create_document(
    body: CreateDocumentRequest,
    usecase: ManageDep,
) -> CreateDocumentResponse:
    doc_id = body.doc_id or str(uuid.uuid4())[:16]
    chunk = RAGChunk(
        meta=RAGDocumentMeta(
            doc_id=doc_id,
            chunk_index=0,
            parent_doc_id=body.parent_doc_id,
            source_path=body.source_path,
            trust_tier=body.trust_tier,
            tags=body.tags,
            extra_meta=body.extra_meta,
            valid_from=body.valid_from,
            valid_to=body.valid_to,
        ),
        content=body.content,
    )
    await usecase.create_document(chunk)
    return CreateDocumentResponse(doc_id=doc_id)


@router.put("/{doc_id}", response_model=UpdateDocumentResponse)
async def update_document(
    doc_id: str,
    body: UpdateDocumentRequest,
    usecase: ManageDep,
) -> UpdateDocumentResponse:
    request = UpdateRequest(
        doc_id=doc_id,
        content=body.content,
        trust_tier=body.trust_tier,
        tags=body.tags,
        extra_meta=body.extra_meta,
        valid_from=body.valid_from,
        valid_to=body.valid_to,
    )
    result_id = await usecase.update_document(request)
    return UpdateDocumentResponse(doc_id=result_id)


@router.delete("/{doc_id}", response_model=DeleteDocumentResponse)
async def delete_document(
    doc_id: str,
    usecase: ManageDep,
) -> DeleteDocumentResponse:
    affected = await usecase.delete_document(doc_id)
    if affected == 0:
        raise HTTPException(status_code=404, detail=f"doc_id '{doc_id}' not found")
    return DeleteDocumentResponse(doc_id=doc_id, affected_chunks=affected)


@router.get("/{doc_id}", response_model=GetDocumentResponse)
async def get_document(
    doc_id: str,
    usecase: ManageDep,
) -> GetDocumentResponse:
    chunks = await usecase.get_document(doc_id)
    if not chunks:
        raise HTTPException(status_code=404, detail=f"doc_id '{doc_id}' not found")
    items = [
        ChunkItem(
            doc_id=c.meta.doc_id,
            chunk_index=c.meta.chunk_index,
            content=c.content,
            trust_tier=c.meta.trust_tier,
            tags=c.meta.tags,
            valid_from=c.meta.valid_from,
            valid_to=c.meta.valid_to,
            source_path=c.meta.source_path,
            recorded_at=c.meta.recorded_at,
        )
        for c in chunks
    ]
    return GetDocumentResponse(doc_id=doc_id, chunks=items)