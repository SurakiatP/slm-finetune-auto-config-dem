from typing import Optional
from .base import BaseInferencer, BasePlayground
from .classification import ClassificationInferencer, ClassificationPlayground

def get_inferencer(task_type: str, base_model_path: str, adapter_path: Optional[str] = None) -> BaseInferencer:
    """
    Factory to retrieve the appropriate task-specific inferencer.
    """
    if task_type == "classification":
        return ClassificationInferencer(base_model_path, adapter_path)
    else:
        raise ValueError(f"Unsupported task type for inference: {task_type}")

def get_playground(task_type: str, inferencer: BaseInferencer, run_id: str) -> BasePlayground:
    """
    Factory to retrieve the appropriate task-specific playground UI.
    """
    if task_type == "classification":
        return ClassificationPlayground(inferencer, run_id)
    else:
        raise ValueError(f"Unsupported task type for playground: {task_type}")
