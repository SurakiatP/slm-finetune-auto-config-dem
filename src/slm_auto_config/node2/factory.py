from .base import BaseSDGEngine
from .classification import ClassificationSDGGenerator

def get_sdg_generator(task_type: str, task_description: str, target_count: int) -> BaseSDGEngine:
    """
    Factory function to retrieve the correct SDG Engine based on the Task Type.
    """
    if task_type == "classification":
        return ClassificationSDGGenerator(
            task_description=task_description,
            target_count=target_count
        )
    # elif task_type == "ner":
    #     return NERSDGGenerator(...)
    else:
        raise ValueError(f"Task type '{task_type}' is not currently supported in Node 2 SDG.")
