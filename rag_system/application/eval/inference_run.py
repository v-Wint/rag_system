from typing import Optional

from rag_system.domain import EvalPrediction, Question, RAGConfig, RAGState, QuestionType
from rag_system.application.inference import get_graph


def run_inference(dataset_name: str, config: RAGConfig, question: Question) -> Optional[EvalPrediction]:
    if EvalPrediction.exists(dataset_name, config, question):
        return None

    graph = get_graph(config)

    result = graph.invoke(RAGState(query=question.user_input))

    retrieved_chunks = list(result.get('retrieved_chunks', []))
    if result.get('schema_used_in_prompt') and result.get('retrieved_schema'):
        retrieved_chunks = [result['retrieved_schema']] + retrieved_chunks

    prediction = EvalPrediction.build(
        dataset_name=dataset_name,
        config=config,
        question=question,
        answer=result['answer'],
        retrieved_chunks=retrieved_chunks,
        classified_question_type=result.get('question_type')
    )
    prediction.upsert()
    return prediction
