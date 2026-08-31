from typing import Callable

from rag_system.domain import DocNode

from .base import split, title_body_split, raw_heading


def _content_leaf(doc_path: list[str], rel_path: list[str], text: str) -> DocNode:
    """Leaf for accumulated small entries that flush under the current section."""
    title = rel_path[-1] if rel_path else ''
    node_rel_path = rel_path if rel_path else ['']
    return DocNode(
        title=title,
        text=text,
        doc_path=doc_path,
        rel_path=node_rel_path,
        abs_path=doc_path + node_rel_path,
        is_content=True,
    )


def _section_leaf(doc_path: list[str], rel_path: list[str], title: str, text: str) -> DocNode:
    node_rel_path = rel_path + [title]
    return DocNode(
        title=title,
        text=text,
        doc_path=doc_path,
        rel_path=node_rel_path,
        abs_path=doc_path + node_rel_path,
    )


def _section_node(
    doc_path: list[str],
    rel_path: list[str],
    title: str,
    heading: str,
    children: list[DocNode],
) -> DocNode:
    node_rel_path = rel_path + [title]
    return DocNode(
        title=title,
        text=heading,
        doc_path=doc_path,
        rel_path=node_rel_path,
        abs_path=doc_path + node_rel_path,
        children=children,
    )


def _build_sections(
    text: str,
    doc_path: list[str],
    rel_path: list[str],
    get_size: Callable[[str], int],
    max_size: int,
) -> list[DocNode]:
    accumulator = ''
    children: list[DocNode] = []

    for c in split(text):
        title, body = title_body_split(c)

        if not body or len(c.strip().splitlines()) <= 3:
            candidate = accumulator + '\n' + c.strip()
            candidate_size = get_size(_content_leaf(doc_path, rel_path, candidate).embedding_text)
            if candidate_size >= max_size:
                children.append(_content_leaf(doc_path, rel_path, accumulator))
                accumulator = c.strip()
            else:
                accumulator = candidate
            continue

        section_rel_path = rel_path + [title]
        section_abs_path = doc_path + section_rel_path
        embedding_text = "Document Location: " + " > ".join(section_abs_path) + "\n\n" + c

        if get_size(embedding_text) < max_size:
            children.append(_section_leaf(doc_path, rel_path, title, c))
        else:
            children.append(_section_node(
                doc_path,
                rel_path,
                title,
                raw_heading(c),
                _build_sections(body, doc_path, section_rel_path, get_size, max_size),
            ))

    if accumulator:
        children.append(_content_leaf(doc_path, rel_path, accumulator))

    return children


def build_doc_subtree(
    text: str,
    doc_path: list[str],
    doc_hash: str,
    get_size: Callable[[str], int],
    max_size: int,
) -> DocNode:
    """Build a rich DocNode subtree for one document.

    The subtree is a chain of doc_path segment nodes whose last node holds the
    section/content leaves produced by the hierarchical splitting logic.
    """
    sections = _build_sections(text, doc_path, [], get_size, max_size) if text.strip() else []

    if not doc_path:
        return DocNode(
            title='',
            text='',
            doc_path=[],
            rel_path=[],
            abs_path=[],
            doc_hash=doc_hash,
            children=sections,
        ).reindex('0', 0)

    root = DocNode(
        title=doc_path[0],
        text=doc_path[0],
        doc_path=doc_path,
        rel_path=[],
        abs_path=[doc_path[0]],
        doc_hash=doc_hash,
        children=[],
    )
    current = root
    for level in doc_path[1:]:
        node = DocNode(
            title=level,
            text=level,
            doc_path=doc_path,
            rel_path=[],
            abs_path=current.abs_path + [level],
            children=[],
        )
        current.children.append(node)
        current = node
    current.children = sections
    return root.reindex('0', 0)
