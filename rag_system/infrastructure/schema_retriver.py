from typing import Optional

from langchain_core.runnables import Runnable

from rag_system.domain import SchemaString
from rag_system.infrastructure import mongo_init


class SchemaRetriever(Runnable):
    _cache: dict[tuple[str, int], Optional[str]] = {}

    def __init__(self, config_slug: str, max_size: int):
        self.config_slug = config_slug
        self.max_size = max_size

    def invoke(self, input = None, config = None, **kwargs) -> Optional[str]:
        key = (self.config_slug, self.max_size)
        if key not in self._cache:
            self._cache[key] = self._fetch_schema()
        return self._cache[key]

    def _fetch_schema(self) -> Optional[str]:
        mongo_init()
        result = SchemaString.find_one(
            (SchemaString.config_slug == self.config_slug) & 
            (SchemaString.max_size == self.max_size)
        ).run()
        if not result:
            return None
        return result.text
