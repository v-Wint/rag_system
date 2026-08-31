from rag_system.domain import DocNode


class TreeIndex:
    """Id → node lookup over a whole-KB DocNode tree."""

    def __init__(self, root: DocNode):
        self.root = root
        self._nodes: dict[str, DocNode] = {}
        self._build(self.root)

    def _build(self, node: DocNode) -> None:
        self._nodes[node.id] = node
        for child in node.children:
            self._build(child)

    def get(self, node_id: str) -> DocNode | None:
        return self._nodes.get(node_id)
