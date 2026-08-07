from typing import Optional
from pydantic import model_validator
from pymongo import IndexModel
from bunnet import Document
from bunnet.operators import Set

from rag_system.utils import get_hash

from rag_system.configs.inference import InferenceConfig
from .inference_result import InferenceResult


class EvalPrediction(Document):
    dataset_name: str
    question_id: str
    question: dict

    config: InferenceConfig
    config_hash: str = ''

    result: InferenceResult

    trace_id: Optional[str]

    class Settings:
        name = "eval_predictions"
        indexes = [
            IndexModel(
                ["dataset_name", "config_hash", "question_id"],
                unique=True
            ),
        ]

    @model_validator(mode="after")
    def compute_config_hash(self):
        if not self.config_hash:
            self.config_hash = get_hash(str(self.config.get_params()))
        return self

    def upsert(self) -> None:
        set_fields = {
            'result': self.result
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
        question_id: str,
        config: InferenceConfig
    ) -> bool:
        config_hash = get_hash(str(config.get_params()))
        existing = cls.find_one(
            cls.dataset_name == dataset_name,
            cls.question_id == question_id,
            cls.config_hash == config_hash
        ).run()
        return existing is not None

    @classmethod
    def load(cls, dataset_name: str, config: InferenceConfig):
        return cls.find(
            cls.dataset_name == dataset_name,
            cls.config_hash == get_hash(str(config.get_params()))
        ).to_list()
