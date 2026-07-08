from pydantic import BaseModel, model_validator
from pymongo import IndexModel
from bunnet import Document
from bunnet.operators import Set
from rag_system.domain import RAGConfig, QuestionType
from rag_system.utils import get_hash

class Question(BaseModel):
    id: str
    question_type: QuestionType
    user_input: str
    reference: str

class EvalPrediction(Document):
    dataset_name: str
    config: RAGConfig
    config_hash: str = ''

    question_id: str
    question_type: str
    user_input: str
    reference: str

    answer: str
    retrieved_chunks: list[str]

    @model_validator(mode="after")
    def compute_config_hash(self):
        if not self.config_hash:
            self.config_hash = get_hash(self.config.model_dump_json())
        return self

    @classmethod
    def build(
        cls,
        dataset_name: str,
        config: RAGConfig,
        question: Question,
        answer: str,
        retrieved_chunks: list[str]
    ) -> "EvalPrediction":
        return cls(
            dataset_name=dataset_name,
            config=config,
            question_id=question.id,
            question_type=question.question_type,
            user_input=question.user_input,
            reference=question.reference,
            answer=answer,
            retrieved_chunks=retrieved_chunks
        )

    def upsert(self) -> None:
        set_fields = {
            "answer": self.answer,
            "retrieved_chunks": self.retrieved_chunks,
        }
        EvalPrediction.find_one(
            EvalPrediction.dataset_name == self.dataset_name,
            EvalPrediction.config_hash == self.config_hash,
            EvalPrediction.question_id == self.question_id,
        ).upsert(Set(set_fields), on_insert=self).run()

    @classmethod
    def exists(
        cls,
        dataset_name: str,
        config: RAGConfig,
        question: Question
    ) -> bool:
        config_hash = get_hash(config.model_dump_json())
        existing = cls.find_one(
            cls.dataset_name == dataset_name,
            cls.config_hash == config_hash,
            cls.question_id == question.id,
        ).run()
        return existing is not None

    class Settings:
        name = "eval_predictions"
        indexes = [
            IndexModel(
                ["dataset_name", "config_hash", "question_id"],
                unique=True,
                name="uniq_dataset_config_question",
            ),
        ]
