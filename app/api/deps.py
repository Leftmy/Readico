from functools import lru_cache

from app.config import settings
from app.services.document_parser import DocumentParser
from app.services.llm_service import LLMService
from app.services.rag_service import RAGService
from app.services.vector_store import QdrantVectorStore


@lru_cache()
def get_vector_store() -> QdrantVectorStore:
    """
    Dependency provider for VectorStore client instance (e.g., Qdrant).
    Cached using lru_cache to reuse the connection pool across requests.
    """
    return QdrantVectorStore(
        host=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT,
        api_key=settings.QDRANT_API_KEY,
        collection_name=settings.QDRANT_COLLECTION_NAME,
    )


@lru_cache()
def get_document_parser() -> DocumentParser:
    """
    Dependency provider for DocumentParser instance used for PDF/text parsing and chunking.
    """
    return DocumentParser(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )


def get_rag_service() -> RAGService:
    """
    Dependency provider for RAGService handling search and document indexing.
    Injects the vector store connection.
    """
    vector_store = get_vector_store()
    return RAGService(vector_store=vector_store)


def get_llm_service() -> LLMService:
    """
    Dependency provider for LLMService managing OpenAI API generation calls.
    """
    return LLMService(
        api_key=settings.OPENAI_API_KEY,
        model_name=settings.LLM_MODEL_NAME,
        temperature=settings.LLM_TEMPERATURE,
    )