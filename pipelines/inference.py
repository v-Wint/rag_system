from zenml import pipeline
from steps.inference import inference_step

from rag_system.configs.inference import InferenceConfig, HierarchicalV1InferenceConfig, RecursiveV1InferenceConfig
from rag_system.configs.chunking import HierarchicalV1Config


@pipeline(enable_cache=False)
def inference_pipeline(config: InferenceConfig, queries: list[str]):
    inference_step(config, queries)



if __name__ == "__main__":
    config = RecursiveV1InferenceConfig().resolve()
    inference_pipeline(
        config,
        ["What did I study during my fourth year, first semester?", "What is the capital of France?"]
    )
