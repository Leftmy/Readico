import pytest
from app.schemas.document import DocumentChunk, ChunkMetadata
from app.services.vector_store import QdrantVectorStore


@pytest.fixture
def vector_store() -> QdrantVectorStore:
    """Fixture initializing an isolated in-memory Qdrant instance for each test."""
    return QdrantVectorStore(
        location=":memory:",
        collection_name="test_documents",
        vector_size=384
    )


@pytest.fixture
def sample_chunks() -> list[DocumentChunk]:
    """Sample document chunks for testing upsert and retrieval."""
    return [
        DocumentChunk(
            chunk_id="doc1_chunk_0",
            content="FastAPI is a modern web framework for Python.",
            metadata=ChunkMetadata(
                document_id="doc1",
                chunk_index=0,
                page_number=1,
                source_filename="fastapi_guide.pdf"
            )
        ),
        DocumentChunk(
            chunk_id="doc1_chunk_1",
            content="Qdrant is a high-performance vector database.",
            metadata=ChunkMetadata(
                document_id="doc1",
                chunk_index=1,
                page_number=2,
                source_filename="fastapi_guide.pdf"
            )
        )
    ]


def test_upsert_and_search_chunks(vector_store: QdrantVectorStore, sample_chunks: list[DocumentChunk]):
    """Verify storing chunks with embeddings and searching returns mapped payload correctly."""
    embeddings = [
        [0.1] * 384,
        [0.9] * 384
    ]

    vector_store.upsert_chunks(chunks=sample_chunks, embeddings=embeddings)

    query_vector = [0.85] * 384
    results = vector_store.search(query_vector=query_vector, limit=2)

    assert len(results) == 2
    top_result = results[0]

    assert top_result["chunk_id"] == "doc1_chunk_1"
    assert top_result["content"] == "Qdrant is a high-performance vector database."
    assert top_result["source_filename"] == "fastapi_guide.pdf"
    assert top_result["page_number"] == 2
    assert top_result["document_id"] == "doc1"
    assert "score" in top_result


def test_search_limit_k(vector_store: QdrantVectorStore, sample_chunks: list[DocumentChunk]):
    """Verify limit parameter restricts the number of returned search results."""
    embeddings = [
        [0.1] * 384,
        [0.2] * 384
    ]
    vector_store.upsert_chunks(chunks=sample_chunks, embeddings=embeddings)

    query_vector = [0.15] * 384
    results = vector_store.search(query_vector=query_vector, limit=1)

    assert len(results) == 1


def test_search_empty_collection(vector_store: QdrantVectorStore):
    """Verify search in an empty vector store returns an empty list without raising errors."""
    query_vector = [0.1] * 384
    results = vector_store.search(query_vector=query_vector, limit=5)

    assert results == []


def test_search_results_ordering(vector_store: QdrantVectorStore, sample_chunks: list[DocumentChunk]):
    """Verify search results are sorted by similarity score in descending order."""
    embeddings = [
        [0.1] * 384,
        [0.9] * 384
    ]
    vector_store.upsert_chunks(chunks=sample_chunks, embeddings=embeddings)

    query_vector = [0.88] * 384
    results = vector_store.search(query_vector=query_vector, limit=2)

    assert len(results) == 2
    assert results[0]["score"] >= results[1]["score"]
    assert results[0]["chunk_id"] == "doc1_chunk_1"


def test_vector_dimension_mismatch(vector_store: QdrantVectorStore, sample_chunks: list[DocumentChunk]):
    """Verify ValueError is raised when embedding dimension does not match store configuration."""
    invalid_embeddings = [
        [0.1] * 128,
        [0.2] * 128
    ]

    with pytest.raises(ValueError, match="Vector dimension mismatch"):
        vector_store.upsert_chunks(chunks=sample_chunks, embeddings=invalid_embeddings)