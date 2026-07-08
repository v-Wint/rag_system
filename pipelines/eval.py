from pathlib import Path
from zenml import pipeline

from rag_system.domain import RAGConfig

from steps.eval import load_questions_step, run_inference_step

@pipeline
def evaluation_pipeline(
    dataset_path: Path | str, 
    config: RAGConfig
):
    questions, dataset_name = load_questions_step(dataset_path)
    _, prediction_finished = run_inference_step(
        dataset_name, config, questions
    )

if __name__ == '__main__':
    config = RAGConfig(
        collection_name="chunks__remnote_v1__hierarchical_v1__intfloat_multilingual_e5_base__510__2048",
        preprocess_model="meta-llama/llama-4-scout-17b-16e-instruct"
    )
    dataset_path = "questions/golden.json"
    evaluation_pipeline(dataset_path, config)
