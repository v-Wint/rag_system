from pydantic import BaseModel
from bunnet import Document as BunnetDocument
from bunnet.operators import Set
from pymongo import IndexModel

class SchemaNode(BaseModel):
    title: str = ''
    children: list['SchemaNode'] = []
    def truncated(self, depth: int) -> 'SchemaNode':
        if depth <= 0:
            return SchemaNode(title=self.title, children=[])
        return SchemaNode(
            title=self.title,
            children=[child.truncated(depth - 1) for child in self.children],
        )

    def height(self) -> int:
        if not self.children:
            return 0
        return 1 + max(child.height() for child in self.children)

    def pruned(self, rounds: int) -> 'SchemaNode':
        return SchemaNode(
            title=self.title,
            children=[
                child.pruned(rounds - 1)
                for child in self.children
                if child.height() >= rounds
            ],
        )

    def _to_str(self, depth=0):
        lines = []
        if self.title:
            lines.append('  ' * depth + '- ' + self.title)
        for child in self.children:
            lines.append(child._to_str(depth + 1))
        return '\n'.join(lines)

    def __str__(self):
        return self._to_str()

SchemaNode.model_rebuild()


def unite_schemas(schemas: list[SchemaNode], root_title: str = 'root') -> SchemaNode :
    root = SchemaNode(title=root_title, children=[])

    def merge_into(target: SchemaNode, source: SchemaNode) -> None:
        match = next((c for c in target.children if c.title == source.title), None)
        if not match:
            target.children.append(source)
            return
        # titles match at this level -> keep walking down instead of nesting a duplicate
        for child in source.children:
            merge_into(match, child)

    for schema in schemas:
        merge_into(root, schema)

    return root


class SchemaText(BunnetDocument):
    root: SchemaNode
    text: str
    collection: str

    class Settings:
        name = "schema_texts"
        indexes = [
            IndexModel("collection", unique=True),
        ]

    def upsert(self):
        SchemaText.find_one(
            SchemaText.collection == self.collection
        ).upsert(
            Set({SchemaText.text: self.text}),
            on_insert=self,
        ).run()
