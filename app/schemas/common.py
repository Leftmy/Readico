from typing import Optional
from pydantic import BaseModel, Field


class ErrorDetailResponse(BaseModel):
    """Generic structured error response schema."""
    detail: str = Field(..., description="Detailed description of the error")
    code: Optional[str] = Field(default=None, description="Internal error code")