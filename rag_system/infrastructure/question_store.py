from pathlib import Path
import json


class QuestionStore:
    def __init__(self, root: str | Path = "questions"):
        if isinstance(root, str):
            root = Path(root).resolve()
        self.root = root

    def load(self, dataset_name: str) -> list[dict]:
        with open(self.root / dataset_name, 'r') as f:
            questions = json.loads(f.read())
        return questions
