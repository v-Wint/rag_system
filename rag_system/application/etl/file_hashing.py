import hashlib


def compute_file_hash(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def compute_files_hashes(paths: list[str]) -> dict[str, str]:
    return {path: compute_file_hash(path) for path in paths}
