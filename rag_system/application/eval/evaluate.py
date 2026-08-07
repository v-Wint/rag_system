from ragas.metrics.collections import SemanticSimilarity, ExactMatch
from ragas.embeddings import HuggingFaceEmbeddings

import pandas as pd


from rag_system.domain.eval_prediction import EvalPrediction
from rag_system.settings import settings


def build_samples(predictions: list[EvalPrediction]) -> list[dict]:
    samples = []
    for prediction in predictions:
        sample_dict = prediction.question
        sample_dict['response'] = prediction.result.answer
        if prediction.result.metadata.get('question_type'):
            sample_dict['question_type_predicted'] = prediction.result.metadata.get('question_type')
        if prediction.result.retrieved_chunks:
            sample_dict['retrieved_contexts'] = prediction.result.retrieved_chunks
        samples.append(sample_dict)
    return samples

def evaluate_predictions(
    predictions: list[EvalPrediction],
) -> pd.DataFrame:

    embeddings = HuggingFaceEmbeddings(
        model=settings.TEXT_EMBEDDING_MODEL_ID, 
        api_key=settings.HUGGINGFACE_ACCESS_TOKEN, 
        device=settings.TEXT_EMBEDDING_DEVICE
    )
    similarity = SemanticSimilarity(embeddings=embeddings)
    exact_match = ExactMatch()

    samples = build_samples(predictions)

    for sample in samples:
        sample['semantic_similarity'] = similarity.score(
            reference=sample['reference'],
            response=sample['response']
        ).value
        if sample.get('question_type_predicted'):
            sample['router_accuracy'] = exact_match.score(
                reference=sample['question_type'],
                response=sample['question_type_predicted']
            ).value

    return pd.DataFrame(samples)
