from zenml import step
import mlflow

from rag_system.configs.inference import InferenceConfig
from rag_system.application.eval import evaluate_predictions
from rag_system.infrastructure import mongo_init
from rag_system.domain import EvalPrediction


@step(enable_cache=False, experiment_tracker='mlflow_tracker')  # type: ignore
def run_evaluation_step(dataset_name: str, config: InferenceConfig):
    mongo_init()
    
    mlflow.log_param("dataset_name", dataset_name)
    mlflow.log_params(config.get_params())
        
    predictions = EvalPrediction.load(dataset_name, config)
    result_df = evaluate_predictions(predictions)

    active_run = mlflow.active_run()
    current_run_id = active_run.info.run_id if active_run else None
    if current_run_id:
        for pred in predictions:
            trace_id = getattr(pred, "trace_id", None)
            if trace_id:
                mlflow.set_trace_tag(trace_id, "mlflow.runId", current_run_id)
    
    aggregated_metrics = {}
    metrics_to_aggregate = ["semantic_similarity", "router_accuracy"]
    
    for metric_col in metrics_to_aggregate:
        if metric_col in result_df.columns:
            aggregated_metrics[f"mean_{metric_col}"] = float(result_df[metric_col].mean())
            
    aggregated_metrics["total_samples"] = len(result_df)

    mlflow.log_metrics(aggregated_metrics)
    
    mlflow.log_table(data=result_df, artifact_file="eval_predictions_table.json")

    return result_df
