from typing import Iterator

from pathlib import Path

from rag_system.constants import SUPPORTED_EXTENSIONS


def crawl_document_paths(data_dir: str | Path) -> list[Path]:
    root = Path(data_dir).resolve()

    if not root.exists():
        raise ValueError(f"Directory does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"Path is not a directory: {root}")

    return list(_crawl(root))


def _crawl(directory: Path) -> Iterator[Path]:
    file_stems = {
        p.stem
        for p in directory.iterdir()
        if p.is_file() and p.suffix in SUPPORTED_EXTENSIONS
    }

    for item in directory.iterdir():
        if item.is_file() and item.suffix in SUPPORTED_EXTENSIONS:
            yield item
        elif item.is_dir() and item.name not in file_stems:
            yield from _crawl(item)

    
if __name__ == '__main__':
    print(crawl_document_paths('./data'))
