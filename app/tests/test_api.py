from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

from app.api.deps import (
    get_document_parser,
    get_llm_service,
    get_rag_service,
    get_vector_store,
)
from app.main import app


@pytest.fixture
def api_client():
    """TestClient instance for FastAPI application."""
    return TestClient(app)


@pytest.fixture
def mock_dependencies():
    """Override FastAPI dependencies with mocks to ensure isolated integration tests."""
    mock_vector_store = MagicMock()
    mock_vector_store.health_check.return_value = True

    mock_rag_service = MagicMock()
    mock_rag_service.index_chunks.return_value = 2
    mock_rag_service.search.return_value = [
        {
            "content": "FastAPI enables high performance API endpoints.",
            "score": 0.92,
            "metadata": {
                "document_id": "doc_123",
                "source_filename": "fastapi_doc.pdf",
                "page_number": 2,
            },
        }
    ]

    mock_llm_service = MagicMock()
    mock_llm_service.generate_answer.return_value = {
        "answer": "FastAPI allows building high performance APIs.",
        "tokens_used": 42,
    }

    mock_parser = MagicMock()
    mock_chunk_1 = MagicMock()
    mock_chunk_2 = MagicMock()
    mock_parser.parse.return_value = [mock_chunk_1, mock_chunk_2]

    # Override dependencies in FastAPI app with mocks
    app.dependency_overrides[get_vector_store] = lambda: mock_vector_store
    app.dependency_overrides[get_rag_service] = lambda: mock_rag_service
    app.dependency_overrides[get_llm_service] = lambda: mock_llm_service
    app.dependency_overrides[get_document_parser] = lambda: mock_parser

    yield {
        "vector_store": mock_vector_store,
        "rag_service": mock_rag_service,
        "llm_service": mock_llm_service,
        "parser": mock_parser,
    }

    app.dependency_overrides.clear()


# 1. Tests for GET /api/v1/health

def test_health_check_success(api_client, mock_dependencies):
    """Verify health check returns status ok and app metadata."""
    response = api_client.get("/api/v1/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "app_name" in data
    assert "environment" in data


def test_health_check_vector_store_unreachable(api_client, mock_dependencies):
    """Verify health check handles vector store failure gracefully."""
    mock_dependencies["vector_store"].health_check.side_effect = Exception("Connection refused")

    response = api_client.get("/api/v1/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


# 2. Tests for POST /api/v1/upload

def test_upload_document_success(api_client, mock_dependencies):
    """Verify successful upload, parsing, and indexing of a document."""
    file_bytes = b"Sample documentation text for testing."
    files = {"file": ("manual.pdf", file_bytes, "application/pdf")}

    response = api_client.post("/api/v1/upload", files=files)

    assert response.status_code == 201
    data = response.json()
    assert "id" in data
    assert data["filename"] == "manual.pdf"
    assert data["total_chunks"] == 2
    assert data["status"] == "indexed"

    mock_dependencies["parser"].parse.assert_called_once()
    mock_dependencies["rag_service"].index_chunks.assert_called_once()


def test_upload_empty_file_validation_error(api_client, mock_dependencies):
    """Verify API rejects empty files with HTTP 400."""
    files = {"file": ("empty.txt", b"", "text/plain")}

    response = api_client.post("/api/v1/upload", files=files)

    assert response.status_code == 400
    assert "Uploaded file is empty" in response.json()["detail"]


def test_upload_file_exceeds_max_size_validation_error(api_client, mock_dependencies):
    """Verify API rejects files exceeding size limit with HTTP 400."""
    oversized_bytes = b"0" * (10 * 1024 * 1024 + 1)
    files = {"file": ("large_doc.pdf", oversized_bytes, "application/pdf")}

    response = api_client.post("/api/v1/upload", files=files)

    assert response.status_code == 400
    assert "exceeds maximum allowed size" in response.json()["detail"].lower()


# 3. Tests for POST /api/v1/chat

def test_chat_query_success(api_client, mock_dependencies):
    """Verify processing query through RAG pipeline and LLM generation."""
    payload = {"query": "How does FastAPI work?", "top_k": 3}

    response = api_client.post("/api/v1/chat", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "How does FastAPI work?"
    assert data["answer"] == "FastAPI allows building high performance APIs."
    assert len(data["citations"]) == 1
    assert data["citations"][0]["filename"] == "fastapi_doc.pdf"

    mock_dependencies["rag_service"].search.assert_called_once_with(
        query="How does FastAPI work?",
        top_k=3,
        document_ids=None,
    )
    mock_dependencies["llm_service"].generate_answer.assert_called_once()


def test_chat_query_empty_message_validation(api_client, mock_dependencies):
    """Verify HTTP 422 when query is empty or whitespace."""
    response_empty = api_client.post("/api/v1/chat", json={"query": ""})
    assert response_empty.status_code == 422

    response_spaces = api_client.post("/api/v1/chat", json={"query": "   "})
    assert response_spaces.status_code == 422
    assert "cannot be empty" in str(response_spaces.json()["detail"]).lower()


def test_chat_search_failure_returns_500(api_client, mock_dependencies):
    """Verify vector search exception converts to HTTP 500 Server Error."""
    mock_dependencies["rag_service"].search.side_effect = Exception("Qdrant connection lost")

    response = api_client.post("/api/v1/chat", json={"query": "What is Python?"})

    assert response.status_code == 500
    assert "Vector search failed" in response.json()["detail"]


def test_chat_llm_generation_failure_returns_500(api_client, mock_dependencies):
    """Verify LLM service failure converts to HTTP 500 Server Error."""
    mock_dependencies["llm_service"].generate_answer.side_effect = Exception("OpenAI API Rate Limit Exceeded")

    response = api_client.post("/api/v1/chat", json={"query": "What is Python?"})

    assert response.status_code == 500
    assert "LLM answer generation failed" in response.json()["detail"]


def test_chat_rate_limit_exceeded(api_client, mock_dependencies):
    """Verify HTTP 429 Too Many Requests when endpoint rate limit is exceeded."""
    payload = {"query": "How does FastAPI work?"}

    responses = [api_client.post("/api/v1/chat", json=payload) for _ in range(12)]

    exceeded_responses = [r for r in responses if r.status_code == 429]
    assert len(exceeded_responses) > 0