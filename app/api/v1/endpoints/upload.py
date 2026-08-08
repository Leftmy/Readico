import logging
import uuid
from datetime import datetime
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.api.deps import get_document_parser, get_rag_service
from app.schemas.document import DocumentResponse
from app.schemas.common import ErrorDetailResponse
from app.services.document_parser import DocumentParser
from app.services.rag_service import RAGService

logger = logging.getLogger(__name__)

router = APIRouter()

# Enforce application-level maximum file size limit (10 MB)
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


@router.post(
    "/upload",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorDetailResponse, "description": "Bad Request (e.g. empty file, oversized file, invalid format)"},
        500: {"model": ErrorDetailResponse, "description": "Internal server error during processing or indexing"},
    },
    summary="Upload, parse, and index a document",
)
async def upload_document(
    file: UploadFile = File(...),
    parser: DocumentParser = Depends(get_document_parser),
    rag_service: RAGService = Depends(get_rag_service),
) -> DocumentResponse:
    """
    Upload a document file, validate size and type, parse into text chunks, 
    and store vector embeddings in the database.
    """
    logger.info("Received file upload request: filename='%s', content_type='%s'", file.filename, file.content_type)

    # 1. Determine file size without reading the entire payload into RAM
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)

    logger.debug("File '%s' size measured: %d bytes", file.filename, file_size)

    if file_size == 0:
        logger.warning("Upload rejected: File '%s' is empty.", file.filename)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    if file_size > MAX_FILE_SIZE_BYTES:
        logger.warning(
            "Upload rejected: File '%s' size (%d bytes) exceeds maximum limit of %d bytes.",
            file.filename,
            file_size,
            MAX_FILE_SIZE_BYTES,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Uploaded file exceeds maximum allowed size of {MAX_FILE_SIZE_MB} MB.",
        )

    document_id = str(uuid.uuid4())

    # 2. Parse file content into chunks
    try:
        content = await file.read()
        chunks = parser.parse(content=content, filename=file.filename)
        logger.info("Successfully parsed file '%s' into %d chunks.", file.filename, len(chunks))
    except ValueError as exc:
        logger.warning("Parsing failed for file '%s': %s", file.filename, exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.error("Unexpected error parsing file '%s': %s", file.filename, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process document content.",
        ) from exc

    # 3. Index chunks into vector store
    try:
        indexed_count = rag_service.index_chunks(document_id=document_id, chunks=chunks)
        logger.info("Successfully indexed %d chunks for document_id '%s'.", indexed_count, document_id)
    except Exception as exc:
        logger.error("Failed to index chunks for document_id '%s': %s", document_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to index document chunks.",
        ) from exc

    return DocumentResponse(
        id=document_id,
        filename=file.filename,
        content_type=file.content_type or "application/octet-stream",
        file_size_bytes=file_size,
        upload_timestamp=datetime.utcnow(),
        status="indexed",
        total_chunks=indexed_count,
        error_message=None,
    )