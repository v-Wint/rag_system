from zenml import step, log_metadata

from rag_system.application.inference import build_graph, RAGState, RAGConfig

@step
def inference_step(config: RAGConfig, query: str) -> dict:
    graph = build_graph(config)
    initial_state = RAGState(query=query)
    result = graph.invoke(initial_state)
    log_metadata(metadata={
        "query_length": len(query),
        "question_type": result.get('question_type', ''),
        "retrieved_chunks": len(result.get('retrieved_chunks', 0)),
        "answer_length": len(result.get('answer', ''))
    }, 
    infer_artifact=True)
    return result
