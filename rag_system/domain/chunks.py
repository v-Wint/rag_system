from abc import ABC, abstractmethod
from typing import Annotated, Union, Literal
from pydantic import BaseModel, Field
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

class HierarchicalChunk(BaseChunk):
    chunk_type: Literal["hierarchical"] = "hierarchical"

    title: str
    abs_path: list[str]
    rel_path: list[str]

    @classmethod
    def from_params(cls, title: str, doc_path: list[str], parent_path: list[str], text: str) -> 'HierarchicalChunk':
        rel_path = parent_path + [title]
        abs_path = doc_path + rel_path

        embedding_text = "Document Location: " + " > ".join(abs_path) + "\n\n" + text

        return cls(
            title=title,
            doc_path=doc_path,
            abs_path=abs_path,
            rel_path=rel_path,
            embedding_text=embedding_text,
            text=text
        )

    def to_document(self) -> Document:
        doc = super().to_document()
        doc.metadata.update({
            "title": self.title,
            "abs_path": "/".join(self.abs_path),
            "rel_path": "/".join(self.rel_path),
        })
        return doc

AnyChunk = Annotated[
    Union[HierarchicalChunk],
    Field(discriminator="chunk_type")
]

class ChunkDocument(BaseModel):
    chunk: AnyChunk
    doc_hash: str

    def to_document(self) -> Document:
        doc = self.chunk.to_document()
        doc.id = str(uuid.UUID(get_hash(self.chunk.embedding_text + self.doc_hash)))
        doc.metadata["doc_hash"] = self.doc_hash
        return doc
