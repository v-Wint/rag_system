from pathlib import Path
from enum import Enum


class FileType(str, Enum):
    MARKDOWN = '.md'


def _extract_md(path: Path) -> str:
    return path.read_text(encoding="utf-8")


ETL_EXTRACTION_REPO = {
    FileType.MARKDOWN: _extract_md
}
def extract_text(path: Path) -> str:
    suffix = str(path.suffix)
    f = ETL_EXTRACTION_REPO.get(FileType(suffix))
    if not f:
        raise ValueError(f"Unsupported extension: {suffix}")
    return f(path)
