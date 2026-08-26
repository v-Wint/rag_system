from zenml import step
import mlflow

import pandas as pd

from rag_system.configs.inference import InferenceConfig
from rag_system.application.eval import evaluate_predictions
from rag_system.infrastructure import mongo_init, QuestionStore
from rag_system.domain import EvalPrediction


@step(enable_cache=False, experiment_tracker='mlflow_tracker')  # type: ignore
def run_evaluation_step(dataset_name: str, config: InferenceConfig):
    mongo_init()

    predictions = EvalPrediction.load(dataset_name, config)
    questions = QuestionStore().load(dataset_name)

    client = mlflow.MlflowClient()
    mlflow.log_param("dataset_name", dataset_name)
    mlflow.log_params(config.get_params())

    active_run = mlflow.active_run()
    current_run_id = active_run.info.run_id if active_run else None
    if current_run_id:
        trace_ids = [getattr(pred, "trace_id", None) for pred in predictions]
        trace_ids = [t for t in trace_ids if t]
        if trace_ids:
            client.link_traces_to_run(trace_ids=trace_ids, run_id=current_run_id)


    result_samples = evaluate_predictions(predictions, questions)
    result_df = pd.DataFrame([
        {**{k: v for k, v in s.items() if k != 'metrics'}, **s['metrics']}
        for s in result_samples
    ])

    aggregated_metrics = {}

    for metric in result_samples[0]['metrics'].keys():
        if metric.endswith('_reason'): continue
        aggregated_metrics[f"mean_{metric}"] = float(result_df[metric].mean())
            
    aggregated_metrics["total_samples"] = len(result_df)

    mlflow.log_metrics(aggregated_metrics)
    
    mlflow.log_table(data=result_df, artifact_file="eval_predictions_table.json")

    mapping = {prediction.question_id: prediction  for prediction in predictions}
    for sample in result_samples:
        prediction = mapping[sample['id']]
        prediction.metrics = sample['metrics']
        prediction.upsert()


    return result_df
