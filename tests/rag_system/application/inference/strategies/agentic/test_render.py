import pytest

from rag_system.domain import DocNode
from rag_system.configs.inference import AgenticV1InferenceConfig
from rag_system.application.inference.strategies.agentic.render import (
    location,
    render_expanded,
    subtree_chars,
)
from rag_system.application.inference.strategies.agentic.tree_index import TreeIndex
from rag_system.application.inference.strategies.agentic.tools import make_expand_tool
from rag_system.application.inference.strategies.agentic.v1.graph import build_graph


def _build_root() -> DocNode:
    physics = DocNode(
        title='Physics',
        text='Physics',
        abs_path=['Physics'],
        children=[
            DocNode(
                title='1D Motion',
                text='# 1D Motion\nMotion in a straight line. Position x(t).',
                abs_path=['Physics', '1D Motion'],
            ),
            DocNode(
                title='Vectors',
                text='# Vectors\nDisplacement, velocity, acceleration.',
                abs_path=['Physics', 'Vectors'],
            ),
        ],
    )
    chemistry = DocNode(
        title='Chemistry',
        text='Chemistry',
        abs_path=['Chemistry'],
        children=[
            DocNode(
                title='Bonds',
                text='# Bonds\nCovalent and ionic bonds.',
                abs_path=['Chemistry', 'Bonds'],
            ),
        ],
    )
    root = DocNode(title='root', text='root', children=[physics, chemistry])
    root.reindex()
    return root


class TestLocation:
    def test_root_location(self):
        root = _build_root()
        assert location(root) == "Root"

    def test_nested_location(self):
        root = _build_root()
        node = TreeIndex(root).get("01")
        assert location(node) == "Physics > Vectors"


class TestSubtreeChars:
    def test_counts_all_descendant_text(self):
        root = _build_root()
        physics = TreeIndex(root).get("0")
        expected = len("Physics") + len("# 1D Motion\nMotion in a straight line. Position x(t).") + len("# Vectors\nDisplacement, velocity, acceleration.")
        assert subtree_chars(physics) == expected


class TestRenderExpanded:
    def test_internal_and_leaf_lines(self):
        root = _build_root()
        out = render_expanded(root, budget=10_000, preview_length=40)
        assert "[0] Physics (children: 2, ~" in out
        assert "[1] Chemistry (children: 1, ~" in out
        assert "truncated" not in out

    def test_leaf_line_has_preview_with_title(self):
        root = _build_root()
        physics = TreeIndex(root).get("0")
        out = render_expanded(physics, budget=10_000, preview_length=40)
        assert "[00] 1D Motion (leaf," in out
        assert "— # 1D Motion Motion in a straight" in out

    def test_budget_truncates_branches(self):
        root = _build_root()
        out = render_expanded(root, budget=20, preview_length=40)
        assert "truncated" in out

    def test_tiny_budget_drops_everything(self):
        root = _build_root()
        out = render_expanded(root, budget=1, preview_length=40)
        assert out == "(truncated — call expand with a specific id to see more)"


class TestExpandTool:
    def test_root_overview(self):
        root = _build_root()
        tool = make_expand_tool(TreeIndex(root), max_expand_size=10_000, preview_length=40)
        result = tool.invoke({"node_ids": []})
        assert result.startswith("### Root\n[0] Physics (children: 2, ~")
        assert "[1] Chemistry" in result

    def test_branch_expansion(self):
        root = _build_root()
        tool = make_expand_tool(TreeIndex(root), max_expand_size=10_000, preview_length=40)
        result = tool.invoke({"node_ids": ["0"]})
        assert result.startswith("### Physics\n[00] 1D Motion (leaf")
        assert "[01] Vectors (leaf" in result

    def test_leaf_returns_full_content(self):
        root = _build_root()
        tool = make_expand_tool(TreeIndex(root), max_expand_size=10_000, preview_length=40)
        result = tool.invoke({"node_ids": ["00"]})
        assert result == "### Physics > 1D Motion\n# 1D Motion\nMotion in a straight line. Position x(t)."

    def test_multiple_ids_batched(self):
        root = _build_root()
        tool = make_expand_tool(TreeIndex(root), max_expand_size=10_000, preview_length=40)
        result = tool.invoke({"node_ids": ["0", "1"]})
        assert result.startswith("### Physics")
        assert "### Chemistry" in result

    def test_unknown_id(self):
        root = _build_root()
        tool = make_expand_tool(TreeIndex(root), max_expand_size=10_000, preview_length=40)
        result = tool.invoke({"node_ids": ["999"]})
        assert result == "[ERROR] Unknown node id: 999"


class TestTreeIndex:
    def test_lookup_by_id(self):
        root = _build_root()
        index = TreeIndex(root)
        assert index.get("").title == "root"
        assert index.get("0").title == "Physics"
        assert index.get("01").title == "Vectors"
        assert index.get("nope") is None


class TestBuildGraph:
    def test_graph_compiles_with_index(self):
        index = TreeIndex(_build_root())
        graph = build_graph(AgenticV1InferenceConfig().resolve(), index=index)
        assert graph is not None
