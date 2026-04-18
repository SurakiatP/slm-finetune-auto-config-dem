from .base import BaseIntake
from .classification import ClassificationIntake

def get_intake(task_type: str, run_id: str) -> BaseIntake:
    """
    Factory function to retrieve the correct Intake Engine based on the Task Type.
    """
    if task_type == "classification":
        return ClassificationIntake(run_id=run_id)
    # elif task_type == "ner":
    #     return NERIntake(run_id=run_id)
    else:
        raise ValueError(f"Task type '{task_type}' is not currently supported in Node 1 (Intake).")
