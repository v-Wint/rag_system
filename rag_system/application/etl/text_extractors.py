from pathlib import Path

def extract_text(path: Path) -> str:
    match path.suffix:
        case ".md":
            return _extract_md(path)
        case _:
            raise ValueError(f"Unsupported extension: {path.suffix}")


def _extract_md(path: Path) -> str:
    return path.read_text(encoding="utf-8")
