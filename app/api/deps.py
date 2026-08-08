from functools import lru_cache
from app.services.reranker_service import RerankerService
from fastapi import Depends

from app.config import settings
from app.services.document_parser import DocumentParser
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService
from app.services.rag_service import RAGService
from app.services.vector_store import QdrantVectorStore


@lru_cache()
def get_vector_store() -> QdrantVectorStore:
    return QdrantVectorStore(
        host=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT,
        api_key=getattr(settings, "QDRANT_API_KEY", None),
        collection_name=settings.QDRANT_COLLECTION_NAME,
        vector_size=384,
    )


@lru_cache()
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService(
        model_name="BAAI/bge-small-en-v1.5",
        dimension=384,
    )

@lru_cache()
def get_reranker_service() -> RerankerService:
    return RerankerService(
        model_name=settings.RERANKER_MODEL_NAME,
    )

@lru_cache()
def get_document_parser() -> DocumentParser:
    """
    Dependency provider for DocumentParser.
    """
    return DocumentParser(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )


def get_rag_service(
    vector_store: QdrantVectorStore = Depends(get_vector_store),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    reranker_service: RerankerService = Depends(get_reranker_service),
) -> RAGService:
    """
    Dependency provider for RAGService.
    """
    return RAGService(
        vector_store=vector_store,
        embedding_service=embedding_service,
        reranker_service=reranker_service
    )


def get_llm_service() -> LLMService:
    return LLMService(
        api_key=settings.LLM_API_KEY,
        model_name=settings.LLM_MODEL_NAME,
        base_url=settings.LLM_BASE_URL,
    )