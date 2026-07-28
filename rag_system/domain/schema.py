from typing import Optional, Sequence
from pydantic import BaseModel
from bunnet import Document as BunnetDocument
from bunnet.operators import Set
from pymongo import IndexModel, UpdateOne
from pymongo.results import BulkWriteResult, DeleteResult


class DocumentNode(BaseModel):
    title: str = ''
    children: list['DocumentNode'] = []

    def truncated(self, depth: int) -> 'DocumentNode':
        if depth <= 0:
            return DocumentNode(title=self.title, children=[])
        return DocumentNode(
            title=self.title,
            children=[child.truncated(depth - 1) for child in self.children],
        )

    def height(self) -> int:
        if not self.children:
            return 0
        return 1 + max(child.height() for child in self.children)

    def _to_str(self, depth=0):
        lines = []
        if self.title:
            lines.append('  ' * depth + '- ' + self.title)
        for child in self.children:
            lines.append(child._to_str(depth + 1))
        return '\n'.join(lines)

    @staticmethod
    def unite(trees: list["DocumentNode"], root_title: str = 'root') -> "DocumentNode" :
        root = DocumentNode(title=root_title, children=[])

        def merge_into(target: DocumentNode, source: DocumentNode) -> None:
            match = next((c for c in target.children if c.title == source.title), None)
            if not match:
                target.children.append(source)
                return
            # titles match at this level -> keep walking down instead of nesting a duplicate
            for child in source.children:
                merge_into(match, child)

        for tree in trees:
            merge_into(root, tree)

        return root

    def __str__(self):
        return self._to_str()


DocumentNode.model_rebuild()


class DocumentTree(BunnetDocument):
    root: DocumentNode
    doc_path: str
    config_slug: str

    class Settings:
        name = "document_trees"
        indexes = [
            IndexModel([("config_slug", 1), ("doc_path", 1)], unique=True)
        ]

    @classmethod
    def load(cls, config_slug: str) -> Sequence['DocumentTree']:
        return cls.find_many(DocumentTree.config_slug == config_slug).run()

    @classmethod
    def bulk_upsert(cls, trees: list["DocumentTree"]) -> BulkWriteResult | None:
        if not trees:
            return

        operations = []
        for tree in trees:
            tree_dict = tree.model_dump(exclude={"id"}) 
            operations.append(
                UpdateOne(
                    filter={"doc_path": tree.doc_path},
                    update={"$set": tree_dict},
                    upsert=True
                )
            )
        return cls.get_motor_collection().bulk_write(operations, ordered=False)

    @classmethod
    def delete_by_paths(cls, config_slug: str, paths: list[str]) -> DeleteResult | None:
        if not paths:
            return

        return cls.get_motor_collection().delete_many(
            {"config_slug": config_slug, "doc_path": {"$in": paths}}
        )


class SchemaString(BunnetDocument):
    text: str
    max_size: int
    config_slug: str

    class Settings:
        name = "schema_strings"
        indexes = [
            IndexModel([("config_slug", 1), ("max_size", 1)], unique=True)
        ]

    @classmethod
    def load(cls, config_slug: str, max_size: int) -> Optional['SchemaString']:
        return cls.find_one(
            SchemaString.config_slug == config_slug,
            SchemaString.max_size == max_size,
        ).run()

    def upsert(self):
        SchemaString.find_one(
            SchemaString.config_slug == self.config_slug,
            SchemaString.max_size == self.max_size,
        ).upsert(
            Set({SchemaString.text: self.text}),
            on_insert=self,
        ).run()
