from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class OumiMessage(BaseModel):
    role: str
    content: str

class OumiExample(BaseModel):
    messages: List[OumiMessage]

class DataSplitReport(BaseModel):
    run_id: str
    task_type: str
    total_count: int
    train_count: int
    val_count: int
    test_count: int
    labels: List[str]
    label_distribution: Dict[str, Dict[str, int]]  # split_name -> {label: count}
