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
            "chunk_id": "chunk_1",
            "content": "FastAPI enables high performance API endpoints.",
            "score": 0.92,
            "metadata": {
                "source_filename": "fastapi_doc.pdf",
                "page_number": 2,
            },
        }
    ]

    mock_llm_service = MagicMock()
    mock_llm_service.generate_answer.return_value = {
        "answer": "FastAPI allows building high performance APIs.",
        "sources": [{"source_filename": "fastapi_doc.pdf", "page_number": 2}],
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

    # Clear dependency overrides after test execution
    app.dependency_overrides.clear()


# 1. Tests for GET /api/health

def test_health_check_success(api_client, mock_dependencies):
    """Verify health check returns status ok when vector store is connected."""
    response = api_client.get("/api/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["vector_store"] == "connected"
    mock_dependencies["vector_store"].health_check.assert_called_once()


def test_health_check_vector_store_unreachable(api_client, mock_dependencies):
    """Verify health check handles vector store failure gracefully."""
    mock_dependencies["vector_store"].health_check.side_effect = Exception("Connection refused")

    response = api_client.get("/api/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["vector_store"] == "unreachable"


# 2. Tests for POST /api/upload

def test_upload_document_success(api_client, mock_dependencies):
    """Verify successful upload, parsing, and indexing of a document."""
    file_bytes = b"Sample documentation text for testing."
    files = {"file": ("manual.pdf", file_bytes, "application/pdf")}

    response = api_client.post("/api/upload", files=files)

    assert response.status_code == 201
    data = response.json()
    assert "document_id" in data
    assert data["filename"] == "manual.pdf"
    assert data["chunks_count"] == 2
    assert "successfully parsed and indexed" in data["message"]

    mock_dependencies["parser"].parse.assert_called_once()
    mock_dependencies["rag_service"].index_chunks.assert_called_once()


def test_upload_empty_file_validation_error(api_client, mock_dependencies):
    """Verify API rejects empty files with HTTP 400."""
    files = {"file": ("empty.txt", b"", "text/plain")}

    response = api_client.post("/api/upload", files=files)

    assert response.status_code == 400
    assert "Uploaded file is empty" in response.json()["detail"]


def test_upload_file_exceeds_max_size_validation_error(api_client, mock_dependencies):
    """Verify API rejects files exceeding the 10 MB size limit with HTTP 400."""
    # 10 MB + 1 byte payload
    oversized_bytes = b"0" * (10 * 1024 * 1024 + 1)
    files = {"file": ("large_doc.pdf", oversized_bytes, "application/pdf")}

    response = api_client.post("/api/upload", files=files)

    assert response.status_code == 400
    assert "exceeds maximum allowed size" in response.json()["detail"].lower()


def test_upload_parser_error_handling(api_client, mock_dependencies):
    """Verify ValueError from parser maps to HTTP 400 Bad Request."""
    mock_dependencies["parser"].parse.side_effect = ValueError("Unsupported file format: .exe")

    files = {"file": ("script.exe", b"binary content", "application/octet-stream")}
    response = api_client.post("/api/upload", files=files)

    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]


def test_upload_indexing_failure_returns_500(api_client, mock_dependencies):
    """Verify internal failure during chunk indexing maps to HTTP 500."""
    mock_dependencies["rag_service"].index_chunks.side_effect = Exception("Qdrant write timeout")

    files = {"file": ("doc.txt", b"Valid content", "text/plain")}
    response = api_client.post("/api/upload", files=files)

    assert response.status_code == 500
    assert "Failed to index document chunks" in response.json()["detail"]


# 3. Tests for POST /api/chat

def test_chat_query_success(api_client, mock_dependencies):
    """Verify processing query through RAG pipeline and LLM generation."""
    payload = {"message": "How does FastAPI work?", "top_k": 3}

    response = api_client.post("/api/chat", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "FastAPI allows building high performance APIs."
    assert len(data["sources"]) == 1
    assert data["sources"][0]["source_filename"] == "fastapi_doc.pdf"
    assert data["sources"][0]["page_number"] == 2

    mock_dependencies["rag_service"].search.assert_called_once_with(
        query="How does FastAPI work?",
        top_k=3
    )
    mock_dependencies["llm_service"].generate_answer.assert_called_once()


def test_chat_query_empty_message_validation(api_client, mock_dependencies):
    """Verify HTTP 400/422 when message is empty or whitespace."""
    response_empty = api_client.post("/api/chat", json={"message": ""})
    assert response_empty.status_code in [400, 422]

    response_spaces = api_client.post("/api/chat", json={"message": "   "})
    assert response_spaces.status_code == 400
    assert "cannot be empty" in response_spaces.json()["detail"]


def test_chat_query_message_too_short_validation(api_client, mock_dependencies):
    """Verify API rejects messages shorter than 3 characters."""
    payload = {"message": "hi"}  # Under minimum length threshold

    response = api_client.post("/api/chat", json=payload)

    assert response.status_code in [400, 422]


def test_chat_query_message_too_long_validation(api_client, mock_dependencies):
    """Verify API rejects messages exceeding 2000 characters."""
    oversized_message = "a" * 2001
    payload = {"message": oversized_message}

    response = api_client.post("/api/chat", json=payload)

    assert response.status_code in [400, 422]


def test_chat_search_failure_returns_500(api_client, mock_dependencies):
    """Verify vector search exception converts to HTTP 500 Server Error."""
    mock_dependencies["rag_service"].search.side_effect = Exception("Qdrant connection lost")

    response = api_client.post("/api/chat", json={"message": "What is Python?"})

    assert response.status_code == 500
    assert "Vector search failed" in response.json()["detail"]


def test_chat_llm_generation_failure_returns_500(api_client, mock_dependencies):
    """Verify LLM service failure converts to HTTP 500 Server Error."""
    mock_dependencies["llm_service"].generate_answer.side_effect = Exception("OpenAI API Rate Limit Exceeded")

    response = api_client.post("/api/chat", json={"message": "What is Python?"})

    assert response.status_code == 500
    assert "LLM answer generation failed" in response.json()["detail"]


# 4. Tests for Rate Limiting (Throttling)

def test_chat_rate_limit_exceeded(api_client, mock_dependencies):
    """Verify HTTP 429 Too Many Requests when endpoint rate limit is exceeded."""
    payload = {"message": "How does FastAPI work?"}

    # Execute requests up to and beyond rate limit threshold
    responses = [api_client.post("/api/chat", json=payload) for _ in range(12)]

    exceeded_responses = [r for r in responses if r.status_code == 429]
    assert len(exceeded_responses) > 0
    assert "rate limit exceeded" in exceeded_responses[-1].text.lower()