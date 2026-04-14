from typing import Any, Dict
from .base import BaseConfigGenerator
from .models import SearchSpace, TuningParamRange

class ClassificationConfigGenerator(BaseConfigGenerator):
    """
    Classification-specific Oumi configuration generator.
    Focuses on Macro F1 as the primary metric.
    """
    def enrich_training_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Add classification-specific metrics and early stopping to the training config."""
        # Target F1 Macro for balanced performance across all document types
        config["training"]["metric_for_best_model"] = "eval_f1_macro"
        config["training"]["greater_is_better"] = True
        
        # Add Early Stopping callback if needed (handled in Oumi trainer params)
        config["training"]["load_best_model_at_end"] = True
        
        return config

    def get_tuning_metric(self) -> str:
        """Optuna will target Macro F1."""
        return "eval_f1_macro"

    def get_tuning_direction(self) -> str:
        return "maximize"

    def get_default_search_space(self) -> SearchSpace:
        """Returns a robust search space for classification SFT."""
        return SearchSpace(params={
            "learning_rate": TuningParamRange(type="loguniform", min=1e-5, max=5e-4),
            "lora_r": TuningParamRange(type="categorical", values=[4, 8, 16, 32]),
            "lora_alpha": TuningParamRange(type="categorical", values=[8, 16, 32, 64]),
            "per_device_train_batch_size": TuningParamRange(type="categorical", values=[1, 2, 4])
        })
