from typing import Optional

from rag_system.domain import EvalPrediction, Question, RAGConfig, RAGState
from rag_system.application.inference import get_graph


def run_inference(dataset_name: str, config: RAGConfig, question: Question) -> Optional[EvalPrediction]:
    if EvalPrediction.exists(dataset_name, config, question):
        return None

    graph = get_graph(config)

    result = graph.invoke(RAGState(query=question.user_input))
    
    retrieved_chunks = (
        ([result['retrieved_schema']] if result.get('retrieved_schema') else [])
        + list(result.get('retrieved_chunks') or [])
    )
    
    prediction = EvalPrediction.build(
        dataset_name=dataset_name,
        config=config,
        question=question,
        answer=result['answer'],
        retrieved_chunks=retrieved_chunks
    )
    prediction.upsert()
    return prediction
