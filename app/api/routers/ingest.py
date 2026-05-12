# app/api/routers/ingest.py
from fastapi import APIRouter

from app.api.deps import IngestDep
from app.api.schemas.ingest_schema import (
    IngestFileRequest,
    IngestFileResponse,
    IngestFolderRequest,
    IngestFolderResponse,
    SyncFolderRequest,
    SyncFolderResponse,
)

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("/folder", response_model=IngestFolderResponse)
async def ingest_folder(
    body: IngestFolderRequest,
    usecase: IngestDep,
) -> IngestFolderResponse:
    count, path = await usecase.ingest_folder(
        folder_path=body.folder_path,
        trust_tier=body.trust_tier,
        tags=body.tags,
        valid_from=body.valid_from,
        valid_to=body.valid_to,
    )
    return IngestFolderResponse(queued_files=count, folder_path=path)


@router.post("/file", response_model=IngestFileResponse)
async def ingest_file(
    body: IngestFileRequest,
    usecase: IngestDep,
) -> IngestFileResponse:
    doc_id = await usecase.ingest_file(
        file_path=body.file_path,
        trust_tier=body.trust_tier,
        tags=body.tags,
        valid_from=body.valid_from,
        valid_to=body.valid_to,
    )
    return IngestFileResponse(doc_id=doc_id, source_path=body.file_path)


@router.post("/sync", response_model=SyncFolderResponse)
async def sync_folder(
    body: SyncFolderRequest,
    usecase: IngestDep,
) -> SyncFolderResponse:
    result = await usecase.sync_folder(folder_path=body.folder_path)
    return SyncFolderResponse(**result)