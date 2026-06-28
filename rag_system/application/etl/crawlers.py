from typing import Iterator
from pathlib import Path
from .text_extractors import FileType

SUPPORTED_EXTENSIONS = [x.value for x in FileType]


def crawl_document_paths(data_dir: str | Path) -> list[Path]:
    root = Path(data_dir).resolve()

    if not root.exists():
        raise ValueError(f"Directory does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"Path is not a directory: {root}")

    return list(_crawl(root))


def _crawl(directory: Path) -> Iterator[Path]:
    files = []
    subdirs = []
    file_stems = set()

    for item in directory.iterdir():
        if item.is_file() and item.suffix in SUPPORTED_EXTENSIONS:
            files.append(item)
            file_stems.add(item.stem)
        elif item.is_dir():
            subdirs.append(item)

    yield from files
    for subdir in subdirs:
        if subdir.name not in file_stems:
            yield from _crawl(subdir)
