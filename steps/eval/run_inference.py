from zenml import step
from loguru import logger
import mlflow

from rag_system.configs.inference import InferenceConfig
from rag_system.infrastructure import QuestionStore, mongo_init
from rag_system.domain import EvalPrediction
from rag_system.application.inference import get_runner


@step(enable_cache=False, experiment_tracker="mlflow_tracker")
def run_inference_step(
    dataset_name: str, 
    config: InferenceConfig, 
) -> None:
    store = QuestionStore()

    questions = store.load(dataset_name)

    runner = get_runner(config)
    runner.setup()

    mongo_init()

    mlflow.langchain.autolog()  # type: ignore

    for question in questions:
        if EvalPrediction.exists(dataset_name, question['id'], config):
            continue

        logger.info(f"Running inference for id={question['id']} query={question['user_input']}")

        with mlflow.tracing.context( # type: ignore
                tags={
                    "strategy": config.strategy,
                    "version": config.version,
                }
            ):
                result = runner.predict(question['user_input'])

        trace_id = mlflow.get_last_active_trace_id()
        prediction = EvalPrediction(
            dataset_name=dataset_name,
            question_id=question['id'],
            config=config,
            result=result,
            trace_id=trace_id
        )
        prediction.upsert()

        if trace_id:
            mlflow.set_trace_tag(trace_id, "question_type", result.metadata.get("question_type", "unknown"))
