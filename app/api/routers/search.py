# app/api/routers/search.py
from fastapi import APIRouter

from app.api.deps import SearchDep
from app.api.schemas.search_schema import SearchAPIRequest, SearchAPIResponse, SearchResultItem
from app.domain.models import SearchRequest

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchAPIResponse)
async def search(
    body: SearchAPIRequest,
    usecase: SearchDep,
) -> SearchAPIResponse:
    request = SearchRequest(
        query=body.query,
        as_of=body.as_of,
        trust_tier_min=body.trust_tier_min,
        tags=body.tags,
        top_k=body.top_k,
        search_mode=body.search_mode,
        rerank=body.rerank,
        hybrid_alpha=body.hybrid_alpha,
        use_parent_context=body.use_parent_context,
    )
    results = await usecase.rag_search(request)
    items = [SearchResultItem(**r.model_dump()) for r in results]
    return SearchAPIResponse(results=items, total=len(items))
