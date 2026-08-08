from unittest.mock import MagicMock
import pytest

from app.schemas.document import DocumentChunk, ChunkMetadata
from app.services.vector_store import QdrantVectorStore
from app.services.embedding_service import EmbeddingService
from app.services.rag_service import RAGService
from app.services.reranker_service import RerankerService


@pytest.fixture
def mock_embedding_service() -> MagicMock:
    """Fixture providing a mocked EmbeddingService."""
    service = MagicMock(spec=EmbeddingService)
    service.embed_text.return_value = [0.1] * 384
    service.embed_batch.return_value = [[0.1] * 384, [0.2] * 384]
    return service


@pytest.fixture
def in_memory_vector_store() -> QdrantVectorStore:
    """Fixture providing an isolated in-memory Qdrant instance."""
    return QdrantVectorStore(
        location=":memory:",
        collection_name="test_rag_collection",
        vector_size=384,
    )


@pytest.fixture
def mock_reranker_service() -> MagicMock:
    """Fixture providing a mocked RerankerService."""
    service = MagicMock(spec=RerankerService)
    service.rerank.return_value = [
        {
            "chunk_id": "doc1_chunk_0",
            "content": "FastAPI is a modern web framework for Python.",
            "score": 0.99,
            "document_id": "doc1",
            "chunk_index": 0,
            "page_number": 1,
            "source_filename": "fastapi.pdf",
        }
    ]
    return service


@pytest.fixture
def rag_service(
    mock_embedding_service: MagicMock,
    in_memory_vector_store: QdrantVectorStore,
    mock_reranker_service: MagicMock,
) -> RAGService:
    """Fixture initializing RAGService with mocked embedding service, in-memory vector store, and reranker."""
    return RAGService(
        embedding_service=mock_embedding_service,
        vector_store=in_memory_vector_store,
        reranker_service=mock_reranker_service,
    )


@pytest.fixture
def sample_chunks() -> list[DocumentChunk]:
    """Sample document chunks for testing."""
    return [
        DocumentChunk(
            chunk_id="doc1_chunk_0",
            content="FastAPI is a modern web framework for Python.",
            metadata=ChunkMetadata(
                document_id="doc1",
                chunk_index=0,
                page_number=1,
                source_filename="fastapi.pdf",
            ),
        ),
        DocumentChunk(
            chunk_id="doc1_chunk_1",
            content="Qdrant provides fast vector search.",
            metadata=ChunkMetadata(
                document_id="doc1",
                chunk_index=1,
                page_number=2,
                source_filename="fastapi.pdf",
            ),
        ),
    ]


# Happy Path Tests

def test_index_chunks_success(
    rag_service: RAGService,
    mock_embedding_service: MagicMock,
    sample_chunks: list[DocumentChunk],
):
    """Verify index_chunks embeds content and stores vectors in Qdrant."""
    indexed_count = rag_service.index_chunks(sample_chunks)

    assert indexed_count == 2
    mock_embedding_service.embed_batch.assert_called_once_with(
        ["FastAPI is a modern web framework for Python.", "Qdrant provides fast vector search."]
    )


def test_search_relevant_chunks(
    rag_service: RAGService,
    mock_embedding_service: MagicMock,
    mock_reranker_service: MagicMock,
    sample_chunks: list[DocumentChunk],
):
    """Verify search converts text query into vector, diverts candidates to the reranker, and returns mapped results."""
    rag_service.index_chunks(sample_chunks)

    mock_embedding_service.embed_text.return_value = [0.1] * 384
    results = rag_service.search(query="Python web framework", top_k=2)

    assert len(results) == 1
    mock_embedding_service.embed_text.assert_called_once_with("Python web framework")
    mock_reranker_service.rerank.assert_called_once()
    assert "chunk_id" in results[0]
    assert "content" in results[0]
    assert "score" in results[0]


def test_search_routes_candidates_to_reranker(
    rag_service: RAGService,
    mock_embedding_service: MagicMock,
    mock_reranker_service: MagicMock,
):
    """Regression test for the reranker integration point within the RAG pipeline."""
    mock_embedding_service.embed_text.return_value = [0.1] * 384

    raw_results = [
        {
            "chunk_id": "doc1_chunk_0",
            "content": "A FastAPI service can return JSON payloads.",
            "document_id": "doc1",
            "chunk_index": 0,
            "page_number": 1,
            "source_filename": "fastapi.pdf",
            "score": 0.61,
        },
        {
            "chunk_id": "doc1_chunk_1",
            "content": "A Qdrant store indexes documents for semantic retrieval.",
            "document_id": "doc1",
            "chunk_index": 1,
            "page_number": 2,
            "source_filename": "fastapi.pdf",
            "score": 0.52,
        },
    ]

    rag_service.vector_store.search = MagicMock(return_value=raw_results)
    reranker_payload = [
        {
            "chunk_id": "doc1_chunk_0",
            "content": "A FastAPI service can return JSON payloads.",
            "document_id": "doc1",
            "chunk_index": 0,
            "page_number": 1,
            "source_filename": "fastapi.pdf",
            "score": 0.91,
        }
    ]
    mock_reranker_service.rerank.return_value = reranker_payload

    results = rag_service.search(query="FastAPI JSON", top_k=1)

    assert results == reranker_payload
    mock_reranker_service.rerank.assert_called_once_with(
        query="FastAPI JSON",
        results=raw_results,
        top_n=1,
        score_threshold=0.35,
    )


def test_index_empty_chunks_list(rag_service: RAGService, mock_embedding_service: MagicMock):
    """Verify indexing an empty list returns 0 without calling embedding service."""
    indexed_count = rag_service.index_chunks([])

    assert indexed_count == 0
    mock_embedding_service.embed_batch.assert_not_called()


def test_search_no_results_found(
    rag_service: RAGService,
    mock_reranker_service: MagicMock,
):
    """Verify searching an empty database returns empty list without errors."""
    mock_reranker_service.rerank.return_value = []
    results = rag_service.search(query="Unknown topic", top_k=5)

    assert results == []


# Fail & Edge Case Tests

def test_search_empty_query_validation(rag_service: RAGService):
    """Verify ValueError is raised when search query is empty or whitespace-only."""
    with pytest.raises(ValueError, match="Search query cannot be empty"):
        rag_service.search(query="", top_k=5)

    with pytest.raises(ValueError, match="Search query cannot be empty"):
        rag_service.search(query="   ", top_k=5)


def test_search_invalid_top_k(rag_service: RAGService):
    """Verify ValueError is raised if top_k parameter is less than or equal to 0."""
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        rag_service.search(query="Python", top_k=0)

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        rag_service.search(query="Python", top_k=-3)


def test_index_chunks_embedding_failure(
    rag_service: RAGService,
    mock_embedding_service: MagicMock,
    sample_chunks: list[DocumentChunk],
):
    """Verify RuntimeError is propagated when EmbeddingService fails during indexing."""
    mock_embedding_service.embed_batch.side_effect = RuntimeError("OpenAI API Failure")

    with pytest.raises(RuntimeError, match="OpenAI API Failure"):
        rag_service.index_chunks(sample_chunks)


def test_search_embedding_failure(rag_service: RAGService, mock_embedding_service: MagicMock):
    """Verify RuntimeError is propagated when EmbeddingService fails during query generation."""
    mock_embedding_service.embed_text.side_effect = RuntimeError("Embedding service unavailable")

    with pytest.raises(RuntimeError, match="Embedding service unavailable"):
        rag_service.search(query="Python framework")