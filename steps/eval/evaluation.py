import mlflow
from zenml import step
from zenml.client import Client
from ragas import evaluate, EvaluationDataset
from ragas.metrics import faithfulness, answer_relevancy
from ragas.run_config import RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from rag_system.settings import settings
from rag_system.domain import RAGConfig
from rag_system.infrastructure import Embedder
from rag_system.application.eval import load_predictions
from langchain_groq import ChatGroq


experiment_tracker = Client().active_stack.experiment_tracker

RAGAS_METRICS = [faithfulness, answer_relevancy]

@step(experiment_tracker=experiment_tracker.name, enable_cache=False)  # type: ignore
def eval_step(dataset_name: str, config: RAGConfig, prediction_finished: bool) -> dict:
    if not prediction_finished:
        return {}

    mlflow_params = {
        "dataset_name": dataset_name,
        **config.model_dump() 
    }
    mlflow.log_params(mlflow_params)

    predictions = load_predictions(dataset_name, config)

    dataset = EvaluationDataset.from_list([
        {
            "user_input": p.user_input,
            "response": p.answer,
            "retrieved_contexts": list(p.retrieved_chunks),
            "reference": p.reference,
        }
        for p in predictions
    ])

    llm = LangchainLLMWrapper(ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", api_key=settings.GROQ_API_KEY)) # type: ignore
    embeddings = LangchainEmbeddingsWrapper(Embedder.from_pretrained(config.embedding_model))

    faithfulness.llm = llm
    faithfulness.embeddings = embeddings
    answer_relevancy.llm = llm 
    answer_relevancy.strictness = 1
    answer_relevancy.embeddings = embeddings

    result = evaluate(
        dataset=dataset,
        metrics=RAGAS_METRICS,
        batch_size=2,
        run_config=RunConfig(max_workers=1, timeout=18000),
    )
    result_df = result.to_pandas() # type: ignore

    agg_metrics = {
        m.name: float(result_df[m.name].replace(0, None).mean()) for m in RAGAS_METRICS
    }
    mlflow.log_metrics(agg_metrics)
    mlflow.log_table(data=result_df, artifact_file="ragas_eval_results.json")

    return agg_metrics
