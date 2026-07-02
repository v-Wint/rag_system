from typing import Callable, Optional
from pathlib import PurePosixPath
import re

from rag_system.domain import Chunk, SchemaNode


def _hierarchical_v1(
        text: str, 
        doc_path: str | list, 
        get_token_count: Callable[[str], int] = len, 
        max_tokens=1_00_000):

    if isinstance(doc_path, str):
        doc_path = doc_path.split('/')

    leaf_node = SchemaNode()
    data_chunks = _split_chunks(text, doc_path, [], get_token_count, max_tokens, leaf_node)
    
    if doc_path:
        root = SchemaNode(title=doc_path[0])
        current = root
        for level in doc_path[1:]:
            new_node = SchemaNode(title=level)
            current.children.append(new_node)
            current = new_node
        current.children.append(leaf_node)
    else:
        root = leaf_node
    
    return data_chunks, root


def _split_chunks(
        text: str, 
        doc_path: list[str], 
        path: list, 
        get_token_count: Callable[[str], int] = len, 
        max_tokens=10_000,
        node: Optional[SchemaNode] = None):

    accumulator = ''
    chunks = split(text)

    result = []

    for c in chunks:
        # chunk-specific
        title, body = title_body_split(c)

        if not body or len(c.strip().splitlines()) <= 3:
            candidate = Chunk.from_params(path[-1] if path else '', doc_path, path[:-1], accumulator + '\n' + c.strip())
            if get_token_count(candidate.embedding_text) >= max_tokens:
                result.append(Chunk.from_params(
                        path[-1] if path else '', 
                        doc_path, 
                        path[:-1], 
                        accumulator)
                )
                accumulator = c.strip()
            else:
                accumulator = candidate.content
            continue

        chunk = Chunk.from_params(title, doc_path, path, c)

        new_node = None
        if node: 
            new_node = SchemaNode(title=title)
            node.children.append(new_node)

        if get_token_count(chunk.embedding_text) < max_tokens:
            result.append(chunk)
        else:
            new_chunks = _split_chunks(body, chunk.doc_path, chunk.rel_path, get_token_count, max_tokens, new_node)
            result += new_chunks

    if accumulator:
        result.append(Chunk.from_params(path[-1] if path else '', doc_path, path[:-1], accumulator))
    return result


def clean_line(line: str) -> str:
    """ Remove markdown artifacts from a line"""
    line = line.lstrip("\ufeff")
    line = re.sub(r"^[\s#\-*+>]+", "", line)     # leading structural junk
    line = re.sub(r"[\*_~`]+", "", line)         # inline markers anywhere
    return line.strip()


def shorten_line(line: str, max_len: int = 50) -> str:
    """Word-aware shortening"""
    words = line.split()
    if not words:
        return ""
    result = []
    total = 0
    for word in words:
        needed = len(word) + (1 if result else 0)
        if total + needed > max_len:
            break
        result.append(word)
        total += needed
    if not result:
        return words[0][:max_len] + "..."
    shortened = " ".join(result)
    return shortened + ("..." if len(result) < len(words) else "")


def title_body_split(text: str) -> tuple[str, str]:
    parts = text.split('\n')
    i = 0
    title = shorten_line(clean_line(parts[0]))
    while len(parts) > i and not title.strip().strip('.'):
        title = shorten_line(clean_line(parts[i]))
        i += 1
    body = '\n'.join(parts[i+1:])
    return title, body

def split_by_bullet(text, max_depth=50):
    current_depth = 1
    bullet = r'[-+*]'
    numbered = r'\d+\.'
    list_item = rf'(?:{bullet}|{numbered})'

    chunks = re.split(rf'\n{list_item} ', text)
    while len(chunks) == 1 and current_depth < max_depth:
        indent = r' ' * current_depth * 4
        chunks = re.split(rf'\n{indent}{list_item} ', text)
        current_depth += 1

    return chunks


def split_by_heading(text, max_depth=7):
    chunks = re.split(r'\n# ', text)
    current_depth = 2
    while len(chunks) == 1 and current_depth < max_depth:
        chunks = re.split(r'\n' + r'#' * current_depth + r' ', text)
        current_depth += 1
    return chunks


def split_by_newlines(text, max_depth=5):
    current_depth = max_depth
    chunks = text.split('\n' * current_depth)
    while len(chunks) == 1 and current_depth > 1:
        current_depth -= 1
        chunks = text.split('\n' * current_depth)
    return chunks


def split(text):
    chunks = split_by_bullet(text)
    if len(chunks) == 1:
        chunks = split_by_heading(text)
    if len(chunks) == 1:
        chunks = split_by_newlines(text)
    if len(chunks) == 1:
        chunks = [c for c in text.split(' ') if c]
    if len(chunks) == 1:
        raise ValueError(f"Could not split: {chunks}")
    return chunks
