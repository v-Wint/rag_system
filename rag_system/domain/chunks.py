from abc import ABC, abstractmethod
from typing import Literal
from pydantic import BaseModel
import uuid
from langchain_core.documents import Document

from rag_system.utils import get_hash


class BaseChunk(BaseModel, ABC):
    text: str
    embedding_text: str
    doc_path: list[str]

    @abstractmethod
    def to_document(self) -> Document:
        return Document(
            page_content=self.embedding_text,
            metadata={
                "text": self.text,
                "doc_path": "/".join(self.doc_path),
            }
        )

class RecursiveChunk(BaseChunk):
    chunk_type: Literal["recursive"] = "recursive"

    def to_document(self) -> Document:
        return super().to_document()


class ChunkDocument(BaseModel):
    chunk: RecursiveChunk
    doc_hash: str

    def to_document(self) -> Document:
        doc = self.chunk.to_document()
        doc.id = str(uuid.UUID(get_hash(self.chunk.embedding_text + self.doc_hash)))
        doc.metadata["doc_hash"] = self.doc_hash
        return doc
