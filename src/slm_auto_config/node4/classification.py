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
        # Note: Advanced metrics like 'metric_for_best_model' are removed for Oumi 0.7 compatibility.
        # Tuning f1_macro is handled at the tuning level in tune.yaml.
        return config

    def get_tuning_metric(self) -> str:
        """Optuna will target eval_loss."""
        return "eval_loss"

    def get_tuning_direction(self) -> str:
        return "minimize"

    def get_default_search_space(self) -> SearchSpace:
        """Returns a robust search space for classification SFT."""
        return SearchSpace(params={
            "learning_rate": TuningParamRange(type="loguniform", min=1e-5, max=5e-4),
            "lora_r": TuningParamRange(type="categorical", values=[4, 8, 16, 32]),
            "lora_alpha": TuningParamRange(type="categorical", values=[8, 16, 32, 64]),
            "per_device_train_batch_size": TuningParamRange(type="categorical", values=[1, 2, 4])
        })
