from typing import List, Dict, Any, Optional
from pydantic import BaseModel, ConfigDict, Field

class IntakeMetadata(BaseModel):
    """
    Summarizes the dataset characteristics post-intake.
    Used by Node 2 (SDG) to formulate diversity rules.
    """
    total_raw_rows: int = 0
    total_valid_rows: int = 0
    total_dropped_rows: int = 0
    unique_labels: List[str] = Field(default_factory=list)
    label_distribution: Dict[str, int] = Field(default_factory=dict)
    
class NormalizedExample(BaseModel):
    """
    The canonical format all Intake pipelines must convert to.
    Unknown/extra fields are preserved in metadata for future use.
    """
    model_config = ConfigDict(extra='allow')
    
    text: str = Field(..., description="The main content or input text")
    label: str = Field(..., description="The target category or classification")
    
