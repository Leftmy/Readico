from fastapi import APIRouter

from app.api.v1.endpoints import chat, health, upload, document

api_v1_router = APIRouter()

api_v1_router.include_router(health.router, tags=["Health"])
api_v1_router.include_router(chat.router, tags=["Chat"])
api_v1_router.include_router(upload.router, tags=["Upload"])
api_v1_router.include_router(document.router, tags=["Documents"])