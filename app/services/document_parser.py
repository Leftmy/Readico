from abc import ABC, abstractmethod
import io
from typing import List, Dict, Type
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.schemas.document import DocumentChunk, ChunkMetadata


# --- 1. Strategy Interface ---

class BaseDocumentParser(ABC):
    """Abstract strategy interface for parsing specific document formats."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    @abstractmethod
    def parse(self, file_bytes: bytes, doc_id: str, filename: str) -> List[DocumentChunk]:
        """Parse raw file bytes into structured document chunks."""
        pass


# --- 2. Concrete Strategies ---

class PDFDocumentParser(BaseDocumentParser):
    """Strategy for extracting text and creating chunks from PDF files."""

    def parse(self, file_bytes: bytes, doc_id: str, filename: str) -> List[DocumentChunk]:
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            chunks: List[DocumentChunk] = []
            chunk_idx = 0

            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if not text.strip():
                    continue

                for chunk_text in self.splitter.split_text(text):
                    chunks.append(
                        DocumentChunk(
                            chunk_id=f"{doc_id}_{chunk_idx}",
                            content=chunk_text,
                            metadata=ChunkMetadata(
                                document_id=doc_id,
                                chunk_index=chunk_idx,
                                page_number=page_num,
                                source_filename=filename
                            )
                        )
                    )
                    chunk_idx += 1

            return chunks
        except Exception as e:
            raise ValueError(f"Failed to parse PDF document: {str(e)}") from e


class TextDocumentParser(BaseDocumentParser):
    """Strategy for parsing plain text (.txt) and Markdown (.md) files."""

    def parse(self, file_bytes: bytes, doc_id: str, filename: str) -> List[DocumentChunk]:
        try:
            text = file_bytes.decode("utf-8")
        except UnicodeDecodeError as e:
            raise ValueError(f"Failed to decode text file as UTF-8: {str(e)}") from e

        if not text.strip():
            return []

        return [
            DocumentChunk(
                chunk_id=f"{doc_id}_{idx}",
                content=chunk_text,
                metadata=ChunkMetadata(
                    document_id=doc_id,
                    chunk_index=idx,
                    page_number=None,
                    source_filename=filename
                )
            )
            for idx, chunk_text in enumerate(self.splitter.split_text(text))
        ]


# --- 3. Factory ---

class DocumentParserFactory:
    """Factory responsible for instantiating the correct parser strategy."""

    _registry: Dict[str, Type[BaseDocumentParser]] = {
        "pdf": PDFDocumentParser,
        "txt": TextDocumentParser,
        "md": TextDocumentParser,
    }

    @classmethod
    def get_parser(
        cls,
        filename: str,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ) -> BaseDocumentParser:
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        parser_cls = cls._registry.get(ext)

        if not parser_cls:
            supported = ", ".join(cls._registry.keys())
            raise ValueError(f"Unsupported file format: '{ext}'. Supported: {supported}")

        return parser_cls(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


# --- 4. Facade Service ---

class DocumentParser:
    """Unified entry-point facade delegating work to Factory & Strategies."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def parse(self, file_bytes: bytes, doc_id: str, filename: str) -> List[DocumentChunk]:
        """Main parsing entry point."""
        parser = DocumentParserFactory.get_parser(
            filename=filename,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        return parser.parse(file_bytes, doc_id, filename)

    def parse_pdf_bytes(self, file_bytes: bytes, doc_id: str, filename: str) -> List[DocumentChunk]:
        return PDFDocumentParser(self.chunk_size, self.chunk_overlap).parse(file_bytes, doc_id, filename)

    def split_text(self, text: str, doc_id: str, filename: str) -> List[DocumentChunk]:
        return TextDocumentParser(self.chunk_size, self.chunk_overlap).parse(text.encode("utf-8"), doc_id, filename)