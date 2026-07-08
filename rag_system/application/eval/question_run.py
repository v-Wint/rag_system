from pathlib import Path
import json

from rag_system.domain import Question

def load_questions(path: str | Path) -> list[Question]:
    with open(path, 'r') as f:
        questions = [Question(**x) for x in json.loads(f.read())]
    return questions
