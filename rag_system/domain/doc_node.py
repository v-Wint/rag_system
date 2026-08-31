import uuid
from typing import Iterator, Optional

from pydantic import BaseModel
from bunnet import Document as BunnetDocument
from bunnet.operators import Set
from langchain_core.documents import Document
from pymongo import IndexModel

from rag_system.configs.enums import SizeMetric
from rag_system.utils import get_hash


class DocNode(BaseModel):
    """Rich, recursive node unifying the hierarchical chunk and the document tree.

    `id` encodes position: a node's id is `parent_id + str(sibling_index)` with
    doc-level roots at "0", "1", ... so "012" means first doc (level 0), second
    child (level 1), third child (level 2). `depth` and `distance_to_leaves` are
    precomputed heuristics for schema pruning and agentic traversal.
    """

    id: str = ''
    title: str = ''
    text: str = ''
    doc_path: list[str] = []
    rel_path: list[str] = []
    abs_path: list[str] = []
    children: list['DocNode'] = []
    depth: int = 0
    distance_to_leaves: int = 0
    doc_hash: Optional[str] = None
    is_content: bool = False

    @property
    def embedding_text(self) -> str:
        return "Document Location: " + " > ".join(self.abs_path) + "\n\n" + self.text

    def to_langchain_document(self, doc_hash: str) -> Document:
        doc = Document(
            page_content=self.embedding_text,
            metadata={
                "text": self.text,
                "doc_path": "/".join(self.doc_path),
                "title": self.title,
                "abs_path": "/".join(self.abs_path),
                "rel_path": "/".join(self.rel_path),
                "doc_hash": doc_hash,
            },
        )
        doc.id = str(uuid.UUID(get_hash(self.embedding_text + doc_hash)))
        return doc

    def walk_leaves(self) -> Iterator['DocNode']:
        if not self.children:
            yield self
            return
        for child in self.children:
            yield from child.walk_leaves()

    def to_title_str(self) -> str:
        lines = []

        def walk(node: 'DocNode', depth: int) -> None:
            if not node.is_content and node.title:
                lines.append('  ' * depth + '- ' + node.title)
            for child in node.children:
                if not child.is_content:
                    walk(child, depth + 1)

        walk(self, 0)
        return '\n'.join(lines)

    def __str__(self) -> str:
        return self.to_title_str()

    def truncated(self, depth: int) -> 'DocNode':
        if depth <= 0:
            return DocNode(title=self.title)
        return DocNode(
            title=self.title,
            children=[child.truncated(depth - 1) for child in self.children],
        )

    def height(self) -> int:
        if not self.children:
            return 0
        return 1 + max(child.height() for child in self.children)

    def _reindex(self, parent_id: str, parent_depth: int, index: int) -> None:
        self.id = parent_id + str(index)
        self.depth = parent_depth + 1
        for i, child in enumerate(self.children):
            child._reindex(self.id, self.depth, i)
        self.distance_to_leaves = 0 if not self.children else 1 + max(
            c.distance_to_leaves for c in self.children
        )

    def reindex(self, root_id: str = '', root_depth: int = -1) -> 'DocNode':
        """Reassign positional ids, depths and leaf-distances from actual positions."""
        self.id = root_id
        self.depth = root_depth
        for i, child in enumerate(self.children):
            child._reindex(self.id, self.depth, i)
        self.distance_to_leaves = 0 if not self.children else 1 + max(
            c.distance_to_leaves for c in self.children
        )
        return self

    @staticmethod
    def unite(subtrees: list['DocNode'], root_title: str = 'root') -> 'DocNode':
        """Merge doc subtrees into a single tree by title, collapsing shared prefixes."""
        root = DocNode(title=root_title, text=root_title)

        def merge_into(target: 'DocNode', source: 'DocNode') -> None:
            if source.is_content:
                return
            match = next(
                (c for c in target.children if not c.is_content and c.title == source.title),
                None,
            )
            if not match:
                target.children.append(source)
                return
            for child in source.children:
                merge_into(match, child)

        for subtree in subtrees:
            merge_into(root, subtree)
        return root.reindex()


DocNode.model_rebuild()


class KBTree(BunnetDocument):
    """The whole knowledge base as a single DocNode, keyed by size metric + size."""

    root: DocNode
    size_metric: SizeMetric
    max_size: int

    class Settings:
        name = "kb_trees"
        indexes = [
            IndexModel([("size_metric", 1), ("max_size", 1)], unique=True)
        ]

    @classmethod
    def load(cls, size_metric: SizeMetric, max_size: int) -> Optional['KBTree']:
        return cls.find_one(
            KBTree.size_metric == size_metric,
            KBTree.max_size == max_size,
        ).run()

    def upsert(self):
        KBTree.find_one(
            KBTree.size_metric == self.size_metric,
            KBTree.max_size == self.max_size,
        ).upsert(
            Set({'root': self.root}),
            on_insert=self,
        ).run()

    def get_doc_hash_map(self) -> dict[str, str]:
        return {
            "/".join(child.doc_path): child.doc_hash
            for child in self.root.children
            if child.doc_path and child.doc_hash is not None
        }
