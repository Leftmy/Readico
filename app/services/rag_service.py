from typing import Any, Dict, List, Optional
from app.schemas.document import DocumentChunk
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import QdrantVectorStore
from app.services.reranker_service import RerankerService


class RAGService:
    """Orchestrator service connecting EmbeddingService and QdrantVectorStore for RAG operations."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: QdrantVectorStore,
        reranker_service: RerankerService
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.reranker_service = reranker_service

    def index_chunks(self, chunks: List[DocumentChunk]) -> int:
        """Embed content of DocumentChunk items and store them in the vector database.
        
        Returns:
            int: The number of successfully indexed chunks.
        """
        if not chunks:
            return 0

        texts = [chunk.content for chunk in chunks]

        embeddings = self.embedding_service.embed_batch(texts)

        self.vector_store.upsert_chunks(chunks=chunks, embeddings=embeddings)

        return len(chunks)

    def search(
    self,
    query: str,
    top_k: int = 4,
    document_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
        """Search vector store and apply reranking to return relevant chunks."""

        if not query or not query.strip():
            raise ValueError("Search query cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        query_vector = self.embedding_service.embed_text(query)

        # Fetch a larger candidate pool from Vector DB (e.g., 15 items)
        candidate_limit = max(top_k * 3, 15)
        raw_results = self.vector_store.search(
            query_vector=query_vector,
            limit=candidate_limit,
            document_ids=document_ids,
        )

        # Pass candidates through Reranker to filter out irrelevant files (e.g., score < 0.35) and return top-k results
        reranked_results = self.reranker_service.rerank(
            query=query,
            results=raw_results,
            top_n=top_k,
            score_threshold=0.35,
        )

        return reranked_results