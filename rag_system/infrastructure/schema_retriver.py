from typing import Optional

from langchain_core.runnables import Runnable

from rag_system.domain import SchemaString
from rag_system.infrastructure import mongo_init
from rag_system.configs.enums import SizeMetric


class SchemaNotFoundError(Exception):
    """Raised when no SchemaString document matches the given key."""
    pass


class SchemaRetriever(Runnable):
    _cache: dict[tuple, str] = {}

    def __init__(
        self,
        size_metric: SizeMetric,
        max_size: int,
        max_schema_size: int,
    ):
        self.size_metric = size_metric
        self.max_size = max_size
        self.max_schema_size = max_schema_size

    def invoke(self, input = None, config = None, **kwargs) -> Optional[str]:
        key = (self.size_metric, self.max_size, self.max_schema_size)
        if key not in self._cache:
            self._cache[key] = self._fetch_schema()
        return self._cache[key]

    def _fetch_schema(self) -> str:
        mongo_init()
        result = SchemaString.find_one(
            SchemaString.size_metric == self.size_metric,
            SchemaString.max_size == self.max_size,
            SchemaString.max_schema_size == self.max_schema_size,
        ).run()
        if not result:
            raise SchemaNotFoundError(
                f"No schema found for size_metric={self.size_metric}, "
                f"max_size={self.max_size}, max_schema_size={self.max_schema_size}"
            )
        return result.text
