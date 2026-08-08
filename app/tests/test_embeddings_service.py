from unittest.mock import MagicMock, patch
import pytest
import numpy as np

from app.services.embedding_service import EmbeddingService


@pytest.fixture
def mock_fastembed():
    """Mock FastEmbed TextEmbedding initialization and embed method."""
    with patch("app.services.embedding_service.TextEmbedding") as mock_class:
        mock_instance = MagicMock()
        mock_class.return_value = mock_instance

        mock_instance.embed.side_effect = lambda texts, batch_size=256: iter(
            [np.array([0.1] * 384) for _ in texts]
        )
        yield mock_class, mock_instance


@pytest.fixture
def embedding_service(mock_fastembed) -> EmbeddingService:
    """Fixture initializing EmbeddingService with mocked FastEmbed."""
    return EmbeddingService(model_name="BAAI/bge-small-en-v1.5")


# Happy Path Tests

def test_embed_single_text(embedding_service: EmbeddingService):
    """Verify single text embedding generation."""
    vector = embedding_service.embed_text("FastAPI guide")
    assert isinstance(vector, list)
    assert len(vector) == 384
    assert vector[0] == 0.1


def test_embed_batch_texts(embedding_service: EmbeddingService):
    """Verify batch text embedding generation."""
    texts = ["FastAPI guide", "Qdrant vector database"]
    vectors = embedding_service.embed_batch(texts)
    assert isinstance(vectors, list)
    assert len(vectors) == 2
    assert len(vectors[0]) == 384
    assert len(vectors[1]) == 384


def test_embed_empty_batch(embedding_service: EmbeddingService, mock_fastembed):
    """Verify empty batch returns empty list without calling model."""
    _, mock_instance = mock_fastembed
    vectors = embedding_service.embed_batch([])
    assert vectors == []
    mock_instance.embed.assert_not_called()


# Validation Tests

def test_embed_empty_text_validation(embedding_service: EmbeddingService):
    """Verify ValueError is raised for empty or whitespace string."""
    with pytest.raises(ValueError, match="Text for embedding cannot be empty"):
        embedding_service.embed_text("")

    with pytest.raises(ValueError, match="Text for embedding cannot be empty"):
        embedding_service.embed_text("   ")


def test_batch_contains_empty_string(embedding_service: EmbeddingService):
    """Verify ValueError is raised if any string in batch is empty."""
    with pytest.raises(ValueError, match="Text for embedding cannot be empty"):
        embedding_service.embed_batch(["Valid text", ""])


# Exception Handling Tests

def test_fastembed_exception(embedding_service: EmbeddingService, mock_fastembed):
    """Verify RuntimeError is raised when FastEmbed raises an exception."""
    _, mock_instance = mock_fastembed
    mock_instance.embed.side_effect = Exception("ONNX Runtime Error")

    with pytest.raises(
        RuntimeError, match="Failed to generate embeddings: ONNX Runtime Error"
    ):
        embedding_service.embed_batch(["Test query"])


def test_embed_large_batch_splits_requests(embedding_service: EmbeddingService, mock_fastembed):
    """Verify batch_size parameter is correctly passed to FastEmbed model."""
    _, mock_instance = mock_fastembed
    texts = [f"Document {i}" for i in range(10)]
    
    embedding_service.embed_batch(texts)
    
    mock_instance.embed.assert_called_once_with(
        texts, batch_size=embedding_service.batch_size
    )