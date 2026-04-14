import pandas as pd
import json
import os
import logging
import glob
from typing import Optional, Dict, Any, List
from .models import TrialResult, EvaluationMetrics, LabelMetric, Node5Metadata

logger = logging.getLogger(__name__)

class MetricsAnalyser:
    """
    Analyses Oumi output files to extract metrics, identify the best trial,
    and prepare metadata for downstream tasks.
    """
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.base_dir = f"runs/{run_id}"
        self.eval_dir = f"{self.base_dir}/evaluation"
        os.makedirs(self.eval_dir, exist_ok=True)

    def analyse_tuning_results(self) -> Node5Metadata:
        """
        Parses the Oumi Optuna results CSV to find the best hyperparameter set.
        """
        csv_path = f"{self.base_dir}/training/output/trials_results.csv"
        if not os.path.exists(csv_path):
            logger.warning(f"Trials results not found at {csv_path}")
            return Node5Metadata()

        try:
            df = pd.read_csv(csv_path)
            # Standard Optuna/Oumi columns: number, state, value, params_...
            trials = []
            for _, row in df.iterrows():
                params = {k.replace('params_', ''): v for k, v in row.items() if k.startswith('params_')}
                trials.append(TrialResult(
                    trial_id=int(row['number']),
                    params=params,
                    metric_value=float(row['value']) if pd.notnull(row['value']) else 0.0,
                    status=row['state']
                ))
            
            # Identify best trial (assuming 'maximize' as per our Node 4 setup)
            best_row = df.loc[df['value'].idxmax()]
            best_trial_id = int(best_row['number'])
            
            metadata = Node5Metadata(
                best_trial_id=best_trial_id,
                # Oumi trial output follows trial_{id} pattern
                best_model_path=f"runs/{self.run_id}/training/output/trial_{best_trial_id}/checkpoint-best",
                all_trials=trials
            )
            
            # Save summary JSON for Node 6
            summary_path = f"{self.eval_dir}/metadata.json"
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(metadata.model_dump(), f, indent=4, ensure_ascii=False)
                
            return metadata
        except Exception as e:
            logger.error(f"Failed to analyse tuning results: {e}")
            return Node5Metadata()

    def parse_evaluation_json(self, eval_json_path: str) -> Optional[EvaluationMetrics]:
        """
        Parses Oumi's evaluation JSON output to extract classification metrics.
        """
        if not os.path.exists(eval_json_path):
            logger.warning(f"Evaluation JSON not found at {eval_json_path}")
            return None

        try:
            with open(eval_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Oumi evaluation output structure extraction
            # This is a simplified mapping; real Oumi might vary based on engine
            metrics = EvaluationMetrics(
                accuracy=data.get("accuracy", 0.0),
                macro_f1=data.get("macro_f1", 0.0),
                label_metrics=data.get("label_metrics", {}),
                confusion_matrix=data.get("confusion_matrix", []),
                labels=data.get("labels", [])
            )
            
            # Save standardized metrics
            with open(f"{self.eval_dir}/metrics.json", 'w', encoding='utf-8') as f:
                json.dump(metrics.model_dump(), f, indent=4, ensure_ascii=False)
                
            return metrics
        except Exception as e:
            logger.error(f"Failed to parse evaluation JSON: {e}")
            return None
