import os
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

class BaseExporter(ABC):
    """
    Abstract base class for model exporters.
    Handles common path management and the export interface.
    """
    def __init__(self, run_id: str, base_model: str, adapter_path: Optional[str] = None):
        self.run_id = run_id
        self.base_model = base_model
        self.adapter_path = adapter_path
        self.export_dir = f"runs/{run_id}/export"
        
        # Create dedicated format subdirectories
        os.makedirs(f"{self.export_dir}/safetensors", exist_ok=True)
        os.makedirs(f"{self.export_dir}/gguf", exist_ok=True)
        os.makedirs(f"{self.export_dir}/onnx", exist_ok=True)

    @abstractmethod
    def export(self, formats: List[str]) -> Dict[str, str]:
        """
        Orchestrates the export process for multiple formats.
        
        Args:
            formats: List of formats to export (e.g., ['safetensors', 'gguf'])
            
        Returns:
            A dictionary mapping format names to their exported paths.
        """
        pass
