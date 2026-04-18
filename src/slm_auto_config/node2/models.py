from typing import List
from pydantic import BaseModel, Field

# --- Pydantic Schema for Judge Output ---
class JudgeOutput(BaseModel):
    fidelity: float = Field(..., ge=0, le=1, description="Score matching the target label")
    naturalness: float = Field(..., ge=0, le=1, description="Score for fluency and realism")
    utility: float = Field(..., ge=0, le=1, description="Score for training value/nuance")
    reasoning: str = Field(..., description="Explanation for the scores")

# --- Meta-Prompting Pydantic Schema ---
class SDGRules(BaseModel):
    diversity_rules: List[str] = Field(..., description="List of at least 8 rules for diversity in generation")
    unknown_diversity_rules: List[str] = Field(..., description="List of at least 5 rules for 'unknown' class data generation")

# --- Generator Output Pydantic Schema ---
class GeneratorBatchOutput(BaseModel):
    results: List[str] = Field(..., description="List of generated diverse candidate strings")
