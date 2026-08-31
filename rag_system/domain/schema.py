from typing import Optional

from bunnet import Document as BunnetDocument
from bunnet.operators import Set
from pymongo import IndexModel

from rag_system.configs.enums import SizeMetric


class SchemaString(BunnetDocument):
    text: str
    size_metric: SizeMetric
    max_size: int
    max_schema_size: int

    class Settings:
        name = "schema_strings"
        indexes = [
            IndexModel(
                [("size_metric", 1), ("max_size", 1), ("max_schema_size", 1)],
                unique=True,
            )
        ]

    @classmethod
    def load(
        cls,
        size_metric: SizeMetric,
        max_size: int,
        max_schema_size: int,
    ) -> Optional['SchemaString']:
        return cls.find_one(
            SchemaString.size_metric == size_metric,
            SchemaString.max_size == max_size,
            SchemaString.max_schema_size == max_schema_size,
        ).run()

    def upsert(self):
        SchemaString.find_one(
            SchemaString.size_metric == self.size_metric,
            SchemaString.max_size == self.max_size,
            SchemaString.max_schema_size == self.max_schema_size,
        ).upsert(
            Set({SchemaString.text: self.text}),
            on_insert=self,
        ).run()
