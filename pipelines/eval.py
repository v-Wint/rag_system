from zenml import pipeline

from rag_system.configs.inference import InferenceConfig, RecursiveV1InferenceConfig, HierarchicalV1InferenceConfig, from_argument
from rag_system.configs.chunking import HierarchicalV1Config

from steps.eval import run_inference_step, run_evaluation_step


@pipeline
def evaluation_pipeline(
    dataset_name: str,
    config: InferenceConfig
):
    run_inference_step(
        dataset_name, config
    )
    run_evaluation_step(
        dataset_name, config, after='run_inference_step'
    )

if __name__ == '__main__':
    config = from_argument()
    if not config:
        config = HierarchicalV1InferenceConfig(
            chunking=HierarchicalV1Config(raw_max_schema_size=3000)
        )
        # config = RecursiveV1InferenceConfig(retrieval=RecursiveV1InferenceConfig.RetrievalConfig(k=4))

    dataset_name = "golden_v1.0.json"
    evaluation_pipeline(dataset_name, config)
