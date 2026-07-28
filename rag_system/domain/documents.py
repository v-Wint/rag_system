from datetime import datetime
from pathlib import Path
from bunnet import Document as BunnetDocument
from pydantic import Field
from pymongo import IndexModel, ASCENDING, UpdateOne
from pymongo.results import BulkWriteResult, DeleteResult

class Document(BunnetDocument):
    root_dir: str
    absolute_path: str
    relative_path: str
    text: str
    hash: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "documents"
        indexes = [
            IndexModel([("absolute_path", ASCENDING)], unique=True),
        ]

    @classmethod
    def from_path(cls, root_dir: Path | str, absolute_path: Path | str) -> "Document":
        from rag_system.utils import get_hash
        from rag_system.application.etl import extract_text

        absolute_path = Path(absolute_path).resolve()
        root = Path(root_dir).resolve()
        text = extract_text(absolute_path)
        hash = get_hash(text)
        return Document(
            root_dir=root.as_posix(),
            absolute_path=absolute_path.as_posix(),
            relative_path=absolute_path.relative_to(root).as_posix(),
            text=text,
            hash=hash,
        )

    @staticmethod
    def get_all_path_hash_pairs() -> dict[str, str]:
        docs = Document.find_all().run()
        return {doc.absolute_path: doc.hash for doc in docs}

    @classmethod
    def bulk_upsert(cls, documents: list["Document"]) -> BulkWriteResult | None:
        if not documents:
            return

        operations = []
        for doc in documents:
            doc_dict = doc.model_dump(exclude={"id"}) 
            operations.append(
                UpdateOne(
                    filter={"absolute_path": doc.absolute_path},
                    update={"$set": doc_dict},
                    upsert=True
                )
            )
        return cls.get_motor_collection().bulk_write(operations, ordered=False)
        

    @classmethod
    def delete_by_paths(cls, absolute_paths: list[str]) -> DeleteResult | None:
        if not absolute_paths:
            return

        return cls.get_motor_collection().delete_many(
            {"absolute_path": {"$in": absolute_paths}}
        )
