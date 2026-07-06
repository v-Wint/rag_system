import mlflow
from zenml import step, log_metadata
from zenml.client import Client
from rag_system.application.inference import build_graph, RAGState, RAGConfig

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name) # type: ignore
def inference_step(config: RAGConfig, query: str) -> dict:
    mlflow.langchain.autolog()  # type: ignore

    graph = build_graph(config)
    initial_state = RAGState(query=query)
    result = graph.invoke(initial_state)

    log_metadata(
        metadata={
            "query_length": len(query),
            "question_type": result.get('question_type', ''),
            "retrieved_chunks": len(result.get('retrieved_chunks', [])),
            "answer_length": len(result.get('answer', '')),
        },
        infer_artifact=True,
    )
    return result
