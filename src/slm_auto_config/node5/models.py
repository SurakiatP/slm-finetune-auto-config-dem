from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

class LabelMetric(BaseModel):
    precision: float
    recall: float
    f1: float
    support: int

class EvaluationMetrics(BaseModel):
    accuracy: float
    macro_f1: float
    label_metrics: Dict[str, LabelMetric]
    confusion_matrix: List[List[int]]
    labels: List[str]

class TrialResult(BaseModel):
    trial_id: int
    params: Dict[str, Any]
    metric_value: float
    status: str

class Node5Metadata(BaseModel):
    best_trial_id: Optional[int] = None
    best_model_path: Optional[str] = None
    all_trials: List[TrialResult] = []
    evaluation: Optional[EvaluationMetrics] = None
