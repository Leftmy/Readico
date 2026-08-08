from unittest.mock import MagicMock, patch
import pytest

from app.services.llm_service import LLMService


@pytest.fixture
def mock_openai_client():
    """Fixture mocking the OpenAI client chat completion response."""
    with patch("app.services.llm_service.OpenAI") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "FastAPI is a modern web framework for Python based on standard type hints."
        mock_response.choices = [mock_choice]
        
        mock_client.chat.completions.create.return_value = mock_response

        yield mock_client


@pytest.fixture
def llm_service(mock_openai_client) -> LLMService:
    """Fixture initializing LLMService with mocked OpenAI client."""
    return LLMService(
        api_key="test-key",
        model_name="gpt-4o-mini"
    )


@pytest.fixture
def sample_context_chunks():
    """Sample context chunks returned from RAG search."""
    return [
        {
            "chunk_id": "doc1_0",
            "content": "FastAPI is a high-performance web framework for Python.",
            "score": 0.89,
            "metadata": {
                "document_id": "doc1",
                "source_filename": "fastapi_guide.pdf",
                "page_number": 3,
                "chunk_index": 0
            }
        },
        {
            "chunk_id": "doc2_1",
            "content": "Qdrant is a vector database used for semantic search.",
            "score": 0.82,
            "metadata": {
                "document_id": "doc2",
                "source_filename": "qdrant_overview.md",
                "page_number": None,
                "chunk_index": 1
            }
        }
    ]


# Happy Path Tests

def test_generate_answer_success(
    llm_service: LLMService,
    mock_openai_client,
    sample_context_chunks
):
    """Verify generating answer formats prompt, calls API and extracts unique sources."""
    result = llm_service.generate_answer(
        query="What is FastAPI?",
        context_chunks=sample_context_chunks
    )

    assert isinstance(result, dict)
    assert "answer" in result
    assert "sources" in result
    assert result["answer"] == "FastAPI is a modern web framework for Python based on standard type hints."
    
    sources = result["sources"]
    assert len(sources) == 2
    assert sources[0]["source_filename"] == "fastapi_guide.pdf"
    assert sources[0]["page_number"] == 3
    assert sources[1]["source_filename"] == "qdrant_overview.md"

    mock_openai_client.chat.completions.create.assert_called_once()


def test_generate_answer_empty_context(llm_service: LLMService, mock_openai_client):
    """Verify LLM handles empty context gracefully without invoking external API."""
    result = llm_service.generate_answer(query="What is Python?", context_chunks=[])

    assert "answer" in result
    assert result["sources"] == []
    assert "Unfortunately the information was not found in the according document" in result["answer"]
    mock_openai_client.chat.completions.create.assert_not_called()


# Fail & Edge Case Tests

def test_generate_answer_empty_query_validation(llm_service: LLMService, sample_context_chunks):
    """Verify ValueError is raised if query is empty or whitespace."""
    with pytest.raises(ValueError, match="Query cannot be empty"):
        llm_service.generate_answer(query="", context_chunks=sample_context_chunks)

    with pytest.raises(ValueError, match="Query cannot be empty"):
        llm_service.generate_answer(query="   ", context_chunks=sample_context_chunks)


def test_missing_api_key():
    """Verify ValueError is raised if LLMService initialized without API key."""
    with pytest.raises(ValueError, match="API key for OpenAI is required"):
        LLMService(api_key=None, model_name="gpt-4o-mini")


def test_openai_chat_api_exception(llm_service: LLMService, mock_openai_client, sample_context_chunks):
    """Verify RuntimeError is raised when OpenAI Chat API fails."""
    mock_openai_client.chat.completions.create.side_effect = Exception("Service unavailable")

    with pytest.raises(RuntimeError, match="Failed to generate answer from LLM: Service unavailable"):
        llm_service.generate_answer(query="What is FastAPI?", context_chunks=sample_context_chunks)