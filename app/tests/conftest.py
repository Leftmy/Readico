import io
import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from pypdf import PdfWriter

from app.main import app


@pytest.fixture
def client() -> TestClient:
    """Provides a FastAPI TestClient instance for API endpoint testing."""
    return TestClient(app)


@pytest.fixture
def mock_pdf_bytes() -> bytes:
    """Generates a valid, minimal in-memory PDF file as bytes."""
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    
    buffer = io.BytesIO()
    writer.write(buffer)
    buffer.seek(0)
    return buffer.getvalue()


@pytest.fixture
def temp_storage_dir(tmp_path: Path) -> Path:
    """Provides an isolated temporary directory for file storage operations."""
    storage_dir = tmp_path / "uploads"
    storage_dir.mkdir(parents=True, exist_ok=True)
    return storage_dir