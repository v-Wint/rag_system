from pathlib import Path

from rag_system.domain import Document
from rag_system.utils import get_hash
from rag_system.application.etl import extract_text

def load_document(root_dir: Path | str, absolute_path: Path | str) -> Document:
    absolute_path = Path(absolute_path).resolve()
    root = Path(root_dir).resolve()
    text = extract_text(absolute_path)
    hash = get_hash(text)
    return Document(
        root_dir=root.as_posix(),
        absolute_path=absolute_path.as_posix(),
        relative_path=absolute_path.relative_to(root).as_posix(),
        text=text,
        hash=hash,
    )
