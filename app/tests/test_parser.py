import pytest
from app.services.document_parser import DocumentParser
from app.schemas.document import DocumentChunk


def test_parse_valid_pdf_bytes(mock_pdf_bytes: bytes):
    """Verify parsing a valid PDF produces a list of DocumentChunk objects with correct metadata."""
    parser = DocumentParser(chunk_size=100, chunk_overlap=20)
    
    doc_id = "doc-uuid-123"
    filename = "test.pdf"
    
    chunks = parser.parse_pdf_bytes(
        file_bytes=mock_pdf_bytes,
        doc_id=doc_id,
        filename=filename
    )
    
    assert isinstance(chunks, list)
    for chunk in chunks:
        assert isinstance(chunk, DocumentChunk)
        assert chunk.metadata.document_id == doc_id
        assert chunk.metadata.source_filename == filename


def test_parse_corrupted_pdf_raises_value_error():
    """Verify corrupted PDF bytes raise an explicit ValueError."""
    parser = DocumentParser()
    corrupted_data = b"Corrupted invalid PDF data"
    
    with pytest.raises(ValueError, match="Failed to parse PDF document"):
        parser.parse_pdf_bytes(
            file_bytes=corrupted_data,
            doc_id="doc-corrupted",
            filename="corrupt.pdf"
        )


def test_text_splitting_overlap_and_indexing():
    """Verify text splitting logic correctly assigns chunk indices, IDs, and overlaps."""
    parser = DocumentParser(chunk_size=50, chunk_overlap=10)
    long_text = (
        "FastAPI is a modern, fast web framework for building APIs with Python. "
        "It provides high performance and automatic OpenAPI documentation generation."
    )
    
    doc_id = "doc-text-1"
    filename = "notes.txt"
    
    chunks = parser.split_text(
        text=long_text,
        doc_id=doc_id,
        filename=filename
    )
    
    assert len(chunks) > 1
    
    for index, chunk in enumerate(chunks):
        assert chunk.chunk_id == f"{doc_id}_{index}"
        assert chunk.metadata.chunk_index == index
        assert chunk.metadata.document_id == doc_id
        assert chunk.metadata.source_filename == filename
        assert len(chunk.content) > 0


def test_empty_text_splitting_returns_empty_list():
    """Verify splitting empty text returns empty list without errors."""
    parser = DocumentParser()
    chunks = parser.split_text(
        text="",
        doc_id="doc-empty",
        filename="empty.txt"
    )
    assert chunks == []


def test_parse_txt_file_content():
    """Verify splitting structured plain .txt file content preserves metadata and text structure."""
    parser = DocumentParser(chunk_size=120, chunk_overlap=20)
    txt_content = (
        "Readico Document Search Service\n\n"
        "This service processes documents using RAG architecture.\n"
        "It splits text into vectorizable chunks for PostgreSQL and Qdrant.\n\n"
        "System requirements:\n"
        "1. Python 3.11+\n"
        "2. Docker & Docker Compose\n"
        "3. PostgreSQL database"
    )
    doc_id = "doc-txt-101"
    filename = "instructions.txt"

    chunks = parser.split_text(text=txt_content, doc_id=doc_id, filename=filename)

    assert len(chunks) >= 2
    assert chunks[0].metadata.source_filename == "instructions.txt"
    assert chunks[0].metadata.document_id == doc_id
    assert "Readico Document Search Service" in chunks[0].content


def test_parse_markdown_file_content():
    """Verify splitting Markdown (.md) content preserves syntax like headers, lists, and code blocks."""
    parser = DocumentParser(chunk_size=150, chunk_overlap=30)
    md_content = (
        "# Readico Architecture\n\n"
        "## Overview\n"
        "Readico is a modern **RAG backend** designed for searching documents.\n\n"
        "## Features\n"
        "- PDF text extraction via `pypdf`\n"
        "- Recursive text splitting\n"
        "- Vector indexing with PostgreSQL / Qdrant\n\n"
        "```python\n"
        "def hello_world():\n"
        "    return 'Readico RAG'\n"
        "```"
    )
    doc_id = "doc-md-202"
    filename = "README.md"

    chunks = parser.split_text(text=md_content, doc_id=doc_id, filename=filename)

    assert len(chunks) > 0
    for index, chunk in enumerate(chunks):
        assert chunk.metadata.source_filename == "README.md"
        assert chunk.metadata.document_id == doc_id
        assert chunk.chunk_id == f"{doc_id}_{index}"

    combined_content = "".join([c.content for c in chunks])
    assert "# Readico Architecture" in combined_content
    assert "```python" in combined_content


def test_unicode_and_ukrainian_text_in_md_and_txt():
    """Verify Ukrainian text and UTF-8 characters are correctly split without breaking strings."""
    parser = DocumentParser(chunk_size=100, chunk_overlap=20)
    ukrainian_md = (
        "# Інструкція з використання Readico\n\n"
        "Readico — це сервіс для пошуку та аналізу документів за допомогою RAG.\n"
        "Він підтримує формати PDF, TXT та Markdown."
    )
    doc_id = "doc-ua-303"
    filename = "документація.md"

    chunks = parser.split_text(text=ukrainian_md, doc_id=doc_id, filename=filename)

    assert len(chunks) > 0
    assert chunks[0].metadata.source_filename == "документація.md"
    has_target_text = any("Readico — це сервіс" in chunk.content for chunk in chunks)
    assert has_target_text, "Target Ukrainian string was not found in any generated chunk"