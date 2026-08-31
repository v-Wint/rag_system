from collections import deque

from rag_system.domain import DocNode


def node_size(node: DocNode) -> int:
    return len(node.text)


def subtree_chars(node: DocNode) -> int:
    total = node_size(node)
    for child in node.children:
        total += subtree_chars(child)
    return total


def location(node: DocNode) -> str:
    return " > ".join(node.abs_path) if node.abs_path else "Root"


def _snippet(text: str, preview_length: int, title: str = '') -> str:
    collapsed = ' '.join(text.split())
    head = collapsed.lstrip('#').strip()
    if title and not head.startswith(title):
        collapsed = f"{title}: {collapsed}"
    if len(collapsed) <= preview_length:
        return collapsed
    return collapsed[:preview_length].rstrip() + "..."


def _render_line(node: DocNode, preview_length: int) -> str:
    if node.children:
        return (
            f"[{node.id}] {node.title} "
            f"(children: {len(node.children)}, ~{subtree_chars(node)} chars)"
        )
    return (
        f"[{node.id}] {node.title} (leaf, {node_size(node)} chars) — "
        f"{_snippet(node.text, preview_length, node.title)}"
    )


def render_expanded(node: DocNode, budget: int, preview_length: int) -> str:
    """BFS render of a node's children bounded by a char budget.

    Nodes are emitted breadth-first; a node is included only if its line fits
    the remaining budget, otherwise its whole subtree is dropped. Appends a
    truncation notice when the budget ran out before the tree was exhausted.
    """
    lines: list[str] = []
    remaining = budget
    queue: deque[DocNode] = deque(c for c in node.children if c.id)
    truncated = False

    while queue and remaining > 0:
        current = queue.popleft()
        line = _render_line(current, preview_length)
        cost = len(line) + 1
        if cost > remaining:
            truncated = True
            continue
        lines.append(line)
        remaining -= cost
        if current.children:
            queue.extend(current.children)

    if truncated or queue:
        lines.append("(truncated — call expand with a specific id to see more)")

    return "\n".join(lines)
