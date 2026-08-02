import mlflow
from zenml import step

from loguru import logger

from rag_system.configs.inference import InferenceConfig
from rag_system.domain import InferenceResult
from rag_system.application.inference import get_runner


@step(experiment_tracker="mlflow_tracker")
def inference_step(config: InferenceConfig, queries: list[str]) -> list[InferenceResult]:
    runner = get_runner(config)
    runner.setup()
    mlflow.langchain.autolog()  # type: ignore
    results = []
    for query in queries:
        logger.info(f"Processing query: {query}")
        with mlflow.tracing.context( # type: ignore
                tags={
                    "strategy": config.strategy,
                    "version": config.version,
                }
            ):
                result = runner.predict(query)

        trace_id = mlflow.get_last_active_trace_id()
        if trace_id:
            mlflow.set_trace_tag(trace_id, "question_type", result.metadata.get("question_type", "unknown"))

        results.append(result)
    return results
