from typing import Optional, Any
from pydantic import BaseModel, Field


class InferenceResult(BaseModel):
    query: str
    retrieved_chunks: Optional[list[str]]
    answer: Optional[str]
    metadata: dict[str, Any] = Field(default_factory=dict)
