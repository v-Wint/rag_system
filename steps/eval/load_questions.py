from typing import Annotated

from pathlib import Path
from zenml import step

from rag_system.domain import Question
from rag_system.application.eval import load_questions

@step(enable_cache=False)
def load_questions_step(
    dataset_path: str | Path
) -> tuple[
    Annotated[list[Question], "questions"], 
    Annotated[str, "dataset_name"]
]:
    dataset_path = Path(dataset_path)
    dataset_name = dataset_path.stem

    questions = load_questions(dataset_path)

    return questions, dataset_name
