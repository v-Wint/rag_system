from typing import Optional

from langchain_core.runnables import Runnable

from rag_system.domain import SchemaText
from rag_system.infrastructure import mongo_init


class SchemaRetriever(Runnable):
    _cache: dict[str, Optional[str]] = {}

    def __init__(self, collection_name: str):
        self.collection_name = collection_name

    def invoke(self, input = None, config = None, **kwargs) -> Optional[str]:
        if self.collection_name not in self._cache:
            self._cache[self.collection_name] = self._fetch_schema()
        return self._cache[self.collection_name]

    def _fetch_schema(self) -> Optional[str]:
        mongo_init()
        result = SchemaText.find_one(SchemaText.collection == self.collection_name).run()
        if not result:
            return None
        return result.text
