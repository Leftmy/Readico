import logging
from fastapi import APIRouter, Depends, HTTPException, Request, status
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.api.deps import get_llm_service, get_rag_service
from app.schemas.rag import Citation, QueryRequest, QueryResponse
from app.services.llm_service import LLMService
from app.services.rag_service import RAGService

logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address)
router = APIRouter()


@router.post(
    "/chat",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Process RAG search query and generate LLM answer",
)
@limiter.limit("10/minute")
async def chat_query(
    request: Request,
    payload: QueryRequest,
    rag_service: RAGService = Depends(get_rag_service),
    llm_service: LLMService = Depends(get_llm_service),
) -> QueryResponse:
    """
    Receive user query, retrieve context chunks from vector database, 
    and generate an answer using the LLM service.
    """
    logger.info(
        "Processing chat query (length=%d, top_k=%d, document_ids=%s)",
        len(payload.query),
        payload.top_k,
        payload.document_ids,
    )

    # 1. Retrieve relevant contexts from RAG Service
    try:
        search_results = rag_service.search(
            query=payload.query,
            top_k=payload.top_k,
            document_ids=payload.document_ids,
        )
        logger.debug("Retrieved %d context chunk(s) from vector store.", len(search_results))
    except Exception as exc:
        logger.error("Vector search failed for query '%s': %s", payload.query, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Vector search failed.",
        ) from exc

    # 2. Generate answer with LLM Service
    try:
        llm_output = llm_service.generate_answer(
            query=payload.query,
            contexts=search_results,
        )
        logger.info("Successfully generated LLM response.")
    except Exception as exc:
        logger.error("LLM answer generation failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="LLM answer generation failed.",
        ) from exc

    # 3. Construct citations list from search results
    citations = [
        Citation(
            document_id=item.get("metadata", {}).get("document_id", "doc_unknown"),
            filename=item.get("metadata", {}).get("source_filename", "unknown"),
            page_number=item.get("metadata", {}).get("page_number"),
            snippet=item.get("content", ""),
            relevance_score=item.get("score"),
        )
        for item in search_results
    ]

    return QueryResponse(
        query=payload.query,
        answer=llm_output.get("answer", ""),
        citations=citations,
        tokens_used=llm_output.get("tokens_used"),
    )