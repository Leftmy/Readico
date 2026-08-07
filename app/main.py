from fastapi import FastAPI
from app.config import settings
from app.api.v1.router import api_v1_router

app = FastAPI(
    title=settings.APP_NAME,
    docs_url="/docs",
    redoc_url="/redoc"
)


app.include_router(api_v1_router, prefix="/api/v1")