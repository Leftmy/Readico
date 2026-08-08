from unittest.mock import MagicMock, patch
import pytest

from app.services.embedding_service import EmbeddingService


@pytest.fixture
def mock_openai_client():
    """Fixture mocking the OpenAI client embeddings response."""
    with patch("app.services.embedding_service.OpenAI") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_item_1 = MagicMock()
        mock_item_1.embedding = [0.1] * 1536
        mock_item_2 = MagicMock()
        mock_item_2.embedding = [0.2] * 1536

        mock_response.data = [mock_item_1, mock_item_2]
        mock_client.embeddings.create.return_value = mock_response

        yield mock_client


@pytest.fixture
def embedding_service(mock_openai_client) -> EmbeddingService:
    """Fixture initializing EmbeddingService with mocked client."""
    return EmbeddingService(
        api_key="test-key",
        model_name="text-embedding-3-small",
        dimension=1536
    )


# Happy Path Tests

def test_embed_single_text(embedding_service: EmbeddingService, mock_openai_client):
    """Verify embedding a single text string returns a float list of correct dimension."""
    text = "FastAPI with Qdrant vector store."
    vector = embedding_service.embed_text(text)

    assert isinstance(vector, list)
    assert len(vector) == 1536
    assert all(isinstance(x, float) for x in vector)
    mock_openai_client.embeddings.create.assert_called_once()


def test_embed_batch_texts(embedding_service: EmbeddingService, mock_openai_client):
    """Verify batch processing converts list of strings into list of vectors."""
    texts = ["Document chunk 1", "Document chunk 2"]
    vectors = embedding_service.embed_batch(texts)

    assert isinstance(vectors, list)
    assert len(vectors) == 2
    assert len(vectors[0]) == 1536
    assert len(vectors[1]) == 1536


def test_embed_empty_batch(embedding_service: EmbeddingService, mock_openai_client):
    """Verify passing an empty list returns empty list without making API calls."""
    vectors = embedding_service.embed_batch([])

    assert vectors == []
    mock_openai_client.embeddings.create.assert_not_called()


# Fail & Edge Case Tests

def test_embed_empty_text_validation(embedding_service: EmbeddingService):
    """Verify ValueError is raised when trying to embed empty or whitespace-only text."""
    with pytest.raises(ValueError, match="Text for embedding cannot be empty"):
        embedding_service.embed_text("")

    with pytest.raises(ValueError, match="Text for embedding cannot be empty"):
        embedding_service.embed_text("   ")


def test_batch_contains_empty_string(embedding_service: EmbeddingService):
    """Verify ValueError is raised if any element in a batch is empty or whitespace."""
    invalid_batch = ["Valid text chunk", "", "Another valid chunk"]

    with pytest.raises(ValueError, match="Text for embedding cannot be empty"):
        embedding_service.embed_batch(invalid_batch)


def test_missing_api_key():
    """Verify ValueError is raised if EmbeddingService is initialized without API key."""
    with pytest.raises(ValueError, match="API key for OpenAI is required"):
        EmbeddingService(api_key=None, model_name="text-embedding-3-small")


def test_openai_api_exception(embedding_service: EmbeddingService, mock_openai_client):
    """Verify RuntimeError is raised when OpenAI API call fails (network/auth/rate limit)."""
    mock_openai_client.embeddings.create.side_effect = Exception("OpenAI API Connection Timeout")

    with pytest.raises(RuntimeError, match="Failed to generate embeddings: OpenAI API Connection Timeout"):
        embedding_service.embed_text("Test query")


def test_dimension_mismatch(embedding_service: EmbeddingService, mock_openai_client):
    """Verify ValueError is raised if OpenAI API returns an embedding with unexpected dimension."""
    mock_response = MagicMock()
    mock_item = MagicMock()
    mock_item.embedding = [0.1] * 512
    mock_response.data = [mock_item]
    mock_openai_client.embeddings.create.return_value = mock_response

    with pytest.raises(ValueError, match="Returned embedding dimension mismatch"):
        embedding_service.embed_text("Test query")

def test_embed_large_batch_splits_requests(mock_openai_client):
    """Verify that a large list of texts is split into sub-batches based on batch_size."""
    service = EmbeddingService(
        api_key="test-key",
        model_name="text-embedding-3-small",
        dimension=1536,
        batch_size=2
    )

    def create_batch_response(model, input):
        """Dynamic mock response matching the size of input chunk."""
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1] * 1536) for _ in range(len(input))
        ]
        return mock_response

    mock_openai_client.embeddings.create.side_effect = create_batch_response

    texts = [f"Chunk {i}" for i in range(5)]
    vectors = service.embed_batch(texts)

    # For 5 elements with batch_size=2, there should be 3 API calls (2 + 2 + 1)
    assert mock_openai_client.embeddings.create.call_count == 3
    assert len(vectors) == 5