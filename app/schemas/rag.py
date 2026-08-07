from typing import List, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Schema for incoming RAG search and retrieval requests."""
    query: str = Field(
        ..., 
        min_length=1, 
        max_length=2000, 
        description="User search prompt or question regarding uploaded documents"
    )
    document_ids: Optional[List[str]] = Field(
        default=None, 
        description="Optional list of specific document IDs to scope the search"
    )
    top_k: int = Field(
        default=4, 
        ge=1, 
        le=20, 
        description="Number of relevant document chunks to retrieve"
    )


class Citation(BaseModel):
    """Schema representing source citation used for RAG response verification."""
    document_id: str = Field(..., description="Referenced document identifier")
    filename: str = Field(..., description="Referenced document filename")
    page_number: Optional[int] = Field(default=None, description="Page number where the snippet was found")
    snippet: str = Field(..., description="Relevant text snippet used for answer generation")
    relevance_score: Optional[float] = Field(default=None, description="Vector similarity / distance score")


class QueryResponse(BaseModel):
    """Schema for final RAG search result with generated answer and citations."""
    query: str = Field(..., description="Original input query")
    answer: str = Field(..., description="LLM-generated answer based on retrieved context")
    citations: List[Citation] = Field(default=[], description="Source citations backing the generated answer")
    tokens_used: Optional[int] = Field(default=None, description="Total tokens consumed during processing")