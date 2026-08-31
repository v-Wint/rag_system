from typing import Annotated

from ragas.metrics.collections import ExactMatch
from ragas.llms import llm_factory
from openai import AsyncOpenAI

from rag_system.domain.eval_prediction import EvalPrediction
from rag_system.settings import settings

from .metrics.answer_correctness import CustomAnswerCorrectness


def build_samples(
    questions: list[dict],
    predictions: list[EvalPrediction]
) -> list[dict]:
    mapping = {prediction.question_id: prediction for prediction in predictions}
    samples = []
    ordered_predictions = []
    for question in questions:
        prediction = mapping[question['id']]
        sample = question.copy()
        sample['response'] = prediction.result.answer
        sample['metrics'] = {}
        if prediction.metrics:
            sample['metrics'] = prediction.metrics.copy()
        if prediction.result.metadata.get('question_type'):
            sample['question_type_predicted'] = prediction.result.metadata.get('question_type')
        if prediction.result.retrieved_chunks:
            sample['retrieved_contexts'] = prediction.result.retrieved_chunks
        samples.append(sample)
        ordered_predictions.append(prediction)
    return samples


def evaluate_predictions(
    predictions: list[EvalPrediction],
    questions: list[dict]
) -> Annotated[list[dict], "samples"]:
    samples = build_samples(questions, predictions)

    exact_match = ExactMatch()

    deepseek_client = AsyncOpenAI(
        api_key=settings.DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com/v1"
    )

    llm = llm_factory(model=settings.LLM_AS_JUDGE_ID, client=deepseek_client, provider='openai', max_tokens=100_000)

    answer_correctness = CustomAnswerCorrectness(llm=llm)

    try:
        for sample in samples:
            if not sample['metrics'].get(answer_correctness.name):
                score = answer_correctness.score(
                    response=sample['response'], reference_facts=sample['reference_facts']
                )
                sample['metrics'][answer_correctness.name] = score.value
                sample['metrics'][answer_correctness.name + "_reason"] = score.reason

            if sample.get('question_type_predicted') and not sample['metrics'].get('router_accuracy'):
                sample['metrics']['router_accuracy'] = exact_match.score(
                    reference=sample['question_type'],
                    response=sample['question_type_predicted']
                ).value
    except (Exception, KeyboardInterrupt) as e:
        print(e)

    return samples
