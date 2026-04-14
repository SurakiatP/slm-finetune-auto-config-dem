import logging
import os
import yaml
from typing import Optional
from slm_auto_config.node4 import get_config_generator, ModelParams, PeftParams, TrainingParams
from slm_auto_config.node5.analyser import MetricsAnalyser
from slm_auto_config.node5.executor import ExecutorGenerator

logger = logging.getLogger(__name__)

class AutoBridge:
    """
    Bridges the outcome of Node 5 (Tuning results) back to Node 4 (Final Training Config).
    """
    def __init__(self, run_id: str, task_type: str, model_name: str):
        self.run_id = run_id
        self.task_type = task_type
        self.model_name = model_name
        self.analyser = MetricsAnalyser(run_id)
        self.executor = ExecutorGenerator(run_id)
        
    def bridge_to_final_run(self, direction: str = "minimize", final_epochs: int = 5):
        """
        1. Identifies the best trial.
        2. Overrides params.
        3. Generates a final training YAML and shell script.
        """
        logger.info(f"🌉 Starting Auto-Bridge for run: {self.run_id}")
        
        # 1. Analyse results
        metadata = self.analyser.analyse_tuning_results(direction=direction)
        if not metadata.best_trial_id is not None:
            logger.error("No best trial found. Cannot bridge to final run.")
            return False
            
        # Extract best params
        # Note: We need to find the trial in metadata.all_trials
        best_trial = next((t for t in metadata.all_trials if t.trial_id == metadata.best_trial_id), None)
        if not best_trial:
            logger.error(f"Could not find details for trial {metadata.best_trial_id}")
            return False
            
        logger.info(f"🏆 Best Trial identified: #{best_trial.trial_id} with score {best_trial.metric_value}")
        
        # 2. Setup Generator and Params
        generator = get_config_generator(self.task_type, self.run_id, self.model_name)
        
        model = ModelParams(model_name=self.model_name)
        peft = PeftParams()
        training = TrainingParams()
        
        # 3. Override with Best Hyperparameters
        params = best_trial.params
        
        # Training Overrides
        if "learning_rate" in params:
            training.learning_rate = float(params["learning_rate"])
        if "per_device_train_batch_size" in params:
            training.per_device_train_batch_size = int(params["per_device_train_batch_size"])
        if "weight_decay" in params:
            training.weight_decay = float(params["weight_decay"])
            
        # PEFT Overrides
        if "lora_r" in params:
            peft.lora_r = int(params["lora_r"])
        if "lora_alpha" in params:
            peft.lora_alpha = int(params["lora_alpha"])
        if "lora_dropout" in params:
            peft.lora_dropout = float(params["lora_dropout"])
            
        # 4. Final Performance Boost: Increase Epochs
        training.num_train_epochs = final_epochs
        
        # 5. Generate FINAL YAML
        # We manually call generator.generate_training_yaml but save it to a different path
        final_config_path = f"runs/{self.run_id}/configs/train_final.yaml"
        # We need to hack Node 4 a bit or just call it and rename the file
        # Actually, let's just make generator.generate_training_yaml return the dict or accept path
        
        # For now, let's call the standard method and manually move/rename if needed
        # or better: we just use the generator's logic here briefly
        
        config_dict = {
            "model": {
                "model_name": model.model_name,
                "trust_remote_code": model.trust_remote_code,
            },
            "data": {
                "train": {
                    "datasets": [{"dataset_name": "text_sft", "dataset_path": f"runs/{self.run_id}/data/train.jsonl"}]
                },
                "validation": {
                    "datasets": [{"dataset_name": "text_sft", "dataset_path": f"runs/{self.run_id}/data/validation.jsonl"}]
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
                "output_dir": f"runs/{self.run_id}/training/final_output",
                "save_final_model": True
            },
            "peft": {
                "q_lora": peft.q_lora,
                "lora_r": peft.lora_r,
                "lora_alpha": peft.lora_alpha,
                "lora_dropout": peft.lora_dropout,
                "lora_target_modules": peft.lora_target_modules
            }
        }
        
        # Enrich for task-specific (classification)
        final_config = generator.enrich_training_config(config_dict)
        
        with open(final_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(final_config, f, sort_keys=False)
            
        logger.info(f"✅ FINAL configuration generated at: {final_config_path}")
        
        # 6. Generate FINAL Script
        self.executor.generate_final_train_script()
        logger.info(f"✅ FINAL training script generated in: {self.executor.script_dir}")
        
        return True
