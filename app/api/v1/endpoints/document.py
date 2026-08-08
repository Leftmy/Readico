from typing import List
from fastapi import APIRouter, Depends, HTTPException, status

from app.schemas.document import DocumentInfo
from app.services.vector_store import QdrantVectorStore
from app.api.deps import get_vector_store

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.get("", response_model=List[DocumentInfo], status_code=status.HTTP_200_OK)
async def get_documents(
    vector_store: QdrantVectorStore = Depends(get_vector_store),
):
    """
    Get a list of all indexed documents in the system.
    """
    try:
        return vector_store.get_all_documents()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving documents: {str(e)}",
        )