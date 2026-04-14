from typing import Optional
from .base import BaseExporter
from .classification import ClassificationExporter

def get_exporter(task_type: str, run_id: str, base_model: str, adapter_path: Optional[str] = None) -> BaseExporter:
    """
    Factory to retrieve the appropriate task-specific exporter.
    """
    if task_type == "classification":
        return ClassificationExporter(run_id, base_model, adapter_path)
    else:
        raise ValueError(f"Unsupported task type for export: {task_type}")
