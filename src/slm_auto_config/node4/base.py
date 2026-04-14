import os
import yaml
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from .models import ModelParams, PeftParams, TrainingParams, SearchSpace

logger = logging.getLogger(__name__)

class BaseConfigGenerator(ABC):
    def __init__(self, run_id: str, model_name: str, task_type: str):
        """
        Base class for generating Oumi configurations.
        
        Args:
            run_id: Unique identifier for the run.
            model_name: Name of the SLM model.
            task_type: Task type (e.g., 'classification').
        """
        self.run_id = run_id
        self.model_name = model_name
        self.task_type = task_type
        # Portable directory structure
        self.base_dir = f"runs/{run_id}"
        self.config_dir = f"{self.base_dir}/configs"
        os.makedirs(self.config_dir, exist_ok=True)

    def generate_training_yaml(self, model: ModelParams, peft: PeftParams, training: TrainingParams) -> str:
        """Generates a standard Oumi train.yaml for manual training."""
        config = {
            "model": {
                "model_name": model.model_name,
                "model_max_length": model.model_max_length,
                "trust_remote_code": model.trust_remote_code,
            },
            "data": {
                "train": {
                    "datasets": [
                        {
                            "dataset_name": "text_sft",
                            "dataset_path": "runs/{}/data/train.jsonl".format(self.run_id),
                        }
                    ]
                },
                "test": {
                    "datasets": [
                        {
                            "dataset_name": "text_sft",
                            "dataset_path": "runs/{}/data/test.jsonl".format(self.run_id),
                        }
                    ]
                },
                "validation": {
                    "datasets": [
                        {
                            "dataset_name": "text_sft",
                            "dataset_path": "runs/{}/data/validation.jsonl".format(self.run_id),
                        }
                    ]
                }
            },
            "training": {
                "trainer_type": training.trainer_type,
                "learning_rate": training.learning_rate,
                "per_device_train_batch_size": training.per_device_train_batch_size,
                "gradient_accumulation_steps": training.gradient_accumulation_steps,
                "num_train_epochs": training.num_train_epochs,
                "warmup_ratio": training.warmup_ratio,
                "weight_decay": training.weight_decay,
                "optimizer": training.optimizer,
                "lr_scheduler_type": training.lr_scheduler_type,
                "logging_steps": training.logging_steps,
                "eval_steps": training.eval_steps,
                "eval_strategy": "steps",
                "output_dir": "runs/{}/training/output".format(self.run_id),
                "save_final_model": True
            },
            "peft": {
                "qlora": peft.qlora,
                "lora_r": peft.lora_r,
                "lora_alpha": peft.lora_alpha,
                "lora_dropout": peft.lora_dropout,
                "lora_target_modules": peft.lora_target_modules
            }
        }
        
        # Enrichment for task-specific fields (e.g., metrics)
        config = self.enrich_training_config(config)
        
        output_path = f"{self.config_dir}/train.yaml"
        self._save_yaml(config, output_path)
        return output_path

    def generate_tuning_yaml(self, model: ModelParams, peft: PeftParams, training: TrainingParams, search_space: SearchSpace) -> str:
        """Generates an Oumi tune.yaml for auto hyperparameter tuning."""
        # Base training config to use as a template for trials
        training_template = self.generate_training_yaml(model, peft, training)
        with open(training_template, 'r', encoding='utf-8') as f:
            template_data = yaml.safe_load(f)

        # Build search spaces for Oumi Tuning (Split between training and peft)
        tunable_training = {}
        tunable_peft = {}
        for name, space in search_space.params.items():
            param_config = {"type": space.type}
            if space.type == "categorical":
                param_config["values"] = space.values
            else:
                param_config["min"] = space.min
                param_config["max"] = space.max
            
            if name in ["learning_rate", "weight_decay", "warmup_ratio", "num_train_epochs", "per_device_train_batch_size"]:
                tunable_training[name] = param_config
            elif name.startswith("lora_"):
                tunable_peft[name] = param_config
            else:
                tunable_training[name] = param_config

        tuning_config = {
            "model": template_data.get("model"),
            "data": template_data.get("data"),
            "tuning": {
                "n_trials": 10,
                "tuner_type": "OPTUNA",
                "tuner_sampler": "TPE",
                "evaluation_metrics": [self.get_tuning_metric()],
                "evaluation_direction": [self.get_tuning_direction()],
                "fixed_training_params": template_data.get("training"),
                "fixed_peft_params": template_data.get("peft"),
                "tunable_training_params": tunable_training,
                "tunable_peft_params": tunable_peft,
                "tuning_study_name": "tuning_{}".format(self.run_id)
            }
        }

        output_path = f"{self.config_dir}/tune.yaml"
        self._save_yaml(tuning_config, output_path)
        return output_path

    @abstractmethod
    def enrich_training_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Override to add task-specific metrics or dataset configurations."""
        pass

    @abstractmethod
    def get_tuning_metric(self) -> str:
        """Return the primary metric for Optuna (e.g., 'eval_loss')."""
        pass

    @abstractmethod
    def get_tuning_direction(self) -> str:
        """Return 'minimize' or 'maximize'."""
        pass

    def _save_yaml(self, data: Dict[str, Any], path: str):
        with open(path, 'w', encoding='utf-8') as f:
            # allow_unicode=True is important for Thai labels/paths if they appear
            yaml.dump(data, f, sort_keys=False, allow_unicode=True)
        logger.info(f"Saved config to {path}")
