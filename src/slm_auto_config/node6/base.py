from abc import ABC, abstractmethod
from typing import Optional, List
from .models import InferenceResponse

class BaseInferencer(ABC):
    """
    Abstract base class for task-specific inferencers.
    """
    def __init__(self, base_model_path: str, adapter_path: Optional[str] = None):
        self.base_model_path = base_model_path
        self.adapter_path = adapter_path

    @abstractmethod
    def predict(self, text: str, **kwargs) -> InferenceResponse:
        """Runs model prediction on the input text."""
        pass

class BasePlayground(ABC):
    """
    Abstract base class for task-specific Gradio interfaces.
    """
    def __init__(self, inferencer: BaseInferencer, run_id: str):
        self.inferencer = inferencer
        self.run_id = run_id

    @abstractmethod
    def launch(self, share: bool = True):
        """Starts the interactive UI."""
        pass
