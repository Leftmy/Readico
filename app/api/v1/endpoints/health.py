import logging
from fastapi import APIRouter, Depends, status

from app.api.deps import get_vector_store
from app.config import settings
from app.schemas.common import HealthCheckResponse
from app.services.vector_store import QdrantVectorStore

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    status_code=status.HTTP_200_OK,
    summary="Check service health status",
)
async def check_health(
    vector_store: QdrantVectorStore = Depends(get_vector_store),
) -> HealthCheckResponse:
    """
    Check the health status of the application and its critical dependencies.
    """
    logger.debug("Health check endpoint triggered.")

    try:
        vector_store.health_check()
        logger.debug("Vector store connectivity verified successfully.")
    except Exception as exc:
        logger.warning("Vector store health check failed: %s", exc)

    return HealthCheckResponse(
        status="ok",
        app_name=getattr(settings, "PROJECT_NAME", "RAG Service API"),
        environment=getattr(settings, "ENVIRONMENT", "development"),
    )