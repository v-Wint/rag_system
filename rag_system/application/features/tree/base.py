import re


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
    while i < len(parts) - 1 and not title.strip().strip('.'):
        i += 1
        title = shorten_line(clean_line(parts[i]))
    body = '\n'.join(parts[i+1:])
    return title, body


def raw_heading(text: str) -> str:
    """Full (non-truncated) cleaned heading of a block, for internal node text."""
    parts = text.split('\n')
    i = 0
    heading = clean_line(parts[0])
    while i < len(parts) - 1 and not heading.strip().strip('.'):
        i += 1
        heading = clean_line(parts[i])
    return heading


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
    if len(chunks) <= 1:
        raise ValueError(f"Could not split: {chunks}")
    return chunks
