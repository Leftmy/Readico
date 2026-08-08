from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class DocumentInfo(BaseModel):
    id: str
    filename: str
    status: str = "Indexed"
    chunks_count: Optional[int] = None

class DocumentBase(BaseModel):
    """Base schema containing essential document metadata."""
    filename: str = Field(..., description="Original name of the uploaded file")
    content_type: str = Field(..., description="MIME type of the file (e.g., application/pdf)")

    # Upper-bound is skipped for this field due to inability to flexibly change file size.
    # Example: 10MB limit is enforced at the application level. 
    # For tests more appropriate approach is to use a smaller size, e.g., 1MB.
    
    file_size_bytes: int = Field(..., ge=0, description="File size in bytes")

class DocumentRequest(DocumentBase):
    """Schema used during document registration/creation."""
    pass


class DocumentResponse(DocumentBase):
    """Schema returned to clients representing stored document status."""
    id: str = Field(..., description="Unique document identifier (UUID)")
    upload_timestamp: datetime = Field(default_factory=datetime.utcnow, description="Upload timestamp")
    status: str = Field(..., description="Processing status: processing | indexed | failed")
    total_chunks: Optional[int] = Field(default=0, description="Total number of generated vector chunks")
    error_message: Optional[str] = Field(default=None, description="Error details if processing failed")

    model_config = ConfigDict(from_attributes=True)

class ChunkMetadata(BaseModel):
    """Metadata attached to an individual text chunk."""
    document_id: str = Field(..., description="Parent document identifier")
    chunk_index: int = Field(..., description="Index sequence number of the chunk")
    page_number: Optional[int] = Field(default=None, description="Page number in the source file, if applicable")
    source_filename: str = Field(..., description="Source filename")


class DocumentChunk(BaseModel):
    """Processed text chunk with associated metadata."""
    chunk_id: str = Field(..., description="Unique chunk identifier")
    content: str = Field(..., description="Extracted text payload of the chunk")
    metadata: ChunkMetadata