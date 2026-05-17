# app/api/routers/generate.py
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.api.deps import GenerateDep
from app.api.schemas.generate_schema import GenerateAPIRequest, GenerateAPIResponse
from app.domain.models import SearchRequest

router = APIRouter(tags=["generate"])


@router.post("/generate", response_model=GenerateAPIResponse)
async def generate(
    body: GenerateAPIRequest,
    usecase: GenerateDep,
) -> GenerateAPIResponse | StreamingResponse:
    search_request = SearchRequest(
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

    if body.stream:
        async def event_stream():
            async for token in usecase.rag_generate_stream(
                search_request,
                system_prompt=body.system_prompt,
                temperature=body.temperature,
                max_tokens=body.max_tokens,
            ):
                yield token

        return StreamingResponse(event_stream(), media_type="text/plain")

    response = await usecase.rag_generate(
        search_request,
        system_prompt=body.system_prompt,
        temperature=body.temperature,
        max_tokens=body.max_tokens,
    )
    return GenerateAPIResponse(
        answer=response.answer,
        sources=response.sources,
        model=response.model,
    )
