from enum import Enum
from pydantic import BaseModel
from typing import Optional

class QuestionType(str, Enum):
    FACT = "fact-question"
    SCHEMA = "schema-question"
    GENERAL = "general-question"

class RAGState(BaseModel):
    query: str
    improved_query: Optional[str] = None
    retrieval_query: Optional[str] = None

    question_type: Optional[QuestionType] = None

    retrieved_schema: Optional[str] = None
    retrieved_chunks: Optional[list[str]] = None

    classification_reasoning: Optional[str] = None
    retrieval_debug: Optional[dict] = None

    answer: Optional[str] = None
