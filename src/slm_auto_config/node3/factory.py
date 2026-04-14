from .base import BaseSplitter
from .classification import ClassificationSplitter

def get_splitter(task_type: str, run_id: str, **kwargs) -> BaseSplitter:
    """
    Factory function to get the appropriate splitter for a task.
    
    Args:
        task_type: Type of task (e.g., 'classification', 'ner')
        run_id: Unique identifier for the run.
        **kwargs: Additional arguments for the splitter (e.g., role, task instructions).
        
    Returns:
        BaseSplitter: An instance of a task-specific splitter.
    """
    if task_type == "classification":
        return ClassificationSplitter(run_id=run_id, **kwargs)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")
