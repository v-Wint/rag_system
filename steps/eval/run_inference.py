from typing import Annotated

from zenml import step, log_metadata
from loguru import logger

from rag_system.domain import EvalPrediction, Question, RAGConfig, RAGState
from rag_system.infrastructure import mongo_init
from rag_system.application.eval import run_inference

@step(enable_cache=False)
def run_inference_step(
    dataset_name: str, 
    config: RAGConfig, 
    questions: list[Question]
) -> tuple[
    Annotated[list[EvalPrediction], "run_predictions"],
    Annotated[bool, "was_completed"]
]:
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
