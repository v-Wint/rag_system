from zenml import pipeline
from steps.inference import inference_step

from rag_system.configs.inference import InferenceConfig, HierarchicalV1InferenceConfig
from rag_system.configs.chunking import HierarchicalV1Config


@pipeline(enable_cache=False)
def inference_pipeline(config: InferenceConfig, queries: list[str]):
    inference_step(config, queries)



if __name__ == "__main__":
    config = HierarchicalV1InferenceConfig(chunking=HierarchicalV1Config(raw_max_schema_size=3000)).resolve()
    inference_pipeline(
        config,
        ["What did I study during my fourth year, first semester?", "What is the capital of France?"]
    )
