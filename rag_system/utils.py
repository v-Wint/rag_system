import hashlib

def get_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def compute_file_hash(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()
