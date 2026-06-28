from pymongo import MongoClient
from bunnet import init_bunnet

from rag_system.settings import settings

_client: MongoClient | None = None

def mongo_init() -> MongoClient:
    global _client
    if _client is None:
        _client = MongoClient(settings.DATABASE_HOST)
        init_bunnet(
            database=_client[settings.DATABASE_NAME],
            document_models=settings.MONGO_DOCUMENT_MODELS
        )
    return _client
