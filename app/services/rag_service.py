from typing import Any, Dict, List
from app.schemas.document import DocumentChunk
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import QdrantVectorStore


class RAGService:
    """Orchestrator service connecting EmbeddingService and QdrantVectorStore for RAG operations."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: QdrantVectorStore,
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store

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

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search top-k most relevant document chunks for a text query.
        
        Args:
            query (str): The search text query.
            top_k (int): Number of top results to retrieve.
            
        Returns:
            List[Dict[str, Any]]: List of matching chunk dictionaries with similarity scores.
        """
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        query_vector = self.embedding_service.embed_text(query)

        results = self.vector_store.search(query_vector=query_vector, limit=top_k)

        return results