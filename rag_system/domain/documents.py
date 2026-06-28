from datetime import datetime
from bunnet import Document as BunnetDocument
from pydantic import Field
from pymongo import IndexModel, ASCENDING

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
