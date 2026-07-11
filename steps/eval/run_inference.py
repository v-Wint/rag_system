from typing import Annotated

from zenml import step, log_metadata
from zenml.client import Client
from loguru import logger
import mlflow

from rag_system.domain import EvalPrediction, Question, RAGConfig, RAGState
from rag_system.infrastructure import mongo_init
from rag_system.application.eval import run_inference

experiment_tracker = Client().active_stack.experiment_tracker

@step(enable_cache=False, experiment_tracker=experiment_tracker.name) # type: ignore
def run_inference_step(
    dataset_name: str, 
    config: RAGConfig, 
    questions: list[Question]
) -> tuple[
    Annotated[list[EvalPrediction], "run_predictions"],
    Annotated[bool, "was_completed"]
]:
    mlflow.langchain.autolog()  # type: ignore
    mongo_init()
    predictions = []
    for question in questions:
        logger.info(f"Running inference for id={question.id} query={question.user_input}")
        try:
            prediction = run_inference(dataset_name, config, question)
        except (Exception, KeyboardInterrupt) as e:
            print(e)
            logger.exception(f"Inference failed for question_id={question.id}")
            return predictions, False
        if prediction is not None:
            predictions.append(prediction)
        else:
            logger.info("Prediction is already run")
    
    log_metadata(metadata={'predictions': len(predictions)}, infer_artifact=True, artifact_name="run_predictions")
    
    return predictions, True
