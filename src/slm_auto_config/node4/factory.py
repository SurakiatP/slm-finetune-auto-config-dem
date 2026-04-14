from .base import BaseConfigGenerator
from .classification import ClassificationConfigGenerator

def get_config_generator(task_type: str, run_id: str, model_name: str) -> BaseConfigGenerator:
    """
    Factory function to get the appropriate configuration generator for a task.
    
    Args:
        task_type: Type of task (e.g., 'classification').
        run_id: Unique identifier for the run.
        model_name: Name of the SLM model.
        
    Returns:
        BaseConfigGenerator: An instance of a task-specific generator.
    """
    if task_type == "classification":
        return ClassificationConfigGenerator(run_id=run_id, model_name=model_name, task_type=task_type)
    else:
        raise ValueError(f"Unsupported task type for config generation: {task_type}")
