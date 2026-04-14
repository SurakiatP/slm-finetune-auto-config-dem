from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union

class InferenceRequest(BaseModel):
    """Data model for an inference request."""
    text: str

class InferenceResponse(BaseModel):
    """Data model for an inference response, including confidence scores."""
    label: str
    confidence: float
    all_scores: Dict[str, float] = Field(default_factory=dict)
    raw_output: str
    error: Optional[str] = None
