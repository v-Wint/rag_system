from pathlib import Path

from rag_system.configs.enums import InferenceStrategy

class PromptStore:
    def __init__(self, root: str | Path = "templates"):
        if isinstance(root, str):
            root = Path(root).resolve()
        self.root = root

    def load_one(self, strategy: InferenceStrategy, version: str, name: str) -> str:
        path = self.root / strategy.value / version / f"{name}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Prompt not found: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            prompt = f.read()
        return prompt
