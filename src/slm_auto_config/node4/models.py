from pydantic import BaseModel, Field
from typing import List, Dict, Union, Optional

class ModelParams(BaseModel):
    model_name: str
    model_max_length: int = 2048
    trust_remote_code: bool = True

class PeftParams(BaseModel):
    q_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = ["q_proj", "v_proj", "k_proj", "o_proj"]

class TrainingParams(BaseModel):
    trainer_type: str = "TRL_SFT"
    learning_rate: float = 1e-4
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    optimizer: str = "paged_adamw_32bit"
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    eval_steps: int = 50
    gradient_checkpointing: bool = True
    save_total_limit: int = 1
    early_stopping_patience: int = 3

class TuningParamRange(BaseModel):
    type: str  # "categorical", "uniform", "loguniform", "int"
    values: Optional[List[Union[str, int, float]]] = None
    min: Optional[float] = None
    max: Optional[float] = None

class SearchSpace(BaseModel):
    # A simplified search space structure for Optuna
    params: Dict[str, TuningParamRange] = Field(default_factory=dict)
