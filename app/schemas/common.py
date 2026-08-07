from typing import Optional
from pydantic import BaseModel, Field


class HealthCheckResponse(BaseModel):
    """Schema for health status endpoint verification."""
    status: str = Field(..., description="System health status")
    app_name: str = Field(..., description="Application name")
    environment: str = Field(..., description="Current running environment")


class ErrorDetailResponse(BaseModel):
    """Generic structured error response schema."""
    detail: str = Field(..., description="Detailed description of the error")
    code: Optional[str] = Field(default=None, description="Internal error code")