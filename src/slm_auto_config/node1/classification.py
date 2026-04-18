import logging
from typing import List, Dict, Any, Tuple

from .base import BaseIntake
from .models import NormalizedExample, IntakeMetadata

logger = logging.getLogger(__name__)

class ClassificationIntake(BaseIntake):
    
    # Heuristics for auto-mapping columns
    TEXT_CANDIDATES = {"text", "content", "document", "sentence", "input", "review", "body"}
    LABEL_CANDIDATES = {"label", "category", "class", "target", "output", "tag"}
    
    def _find_key(self, keys: List[str], candidates: set) -> str:
        """Finds the best matching key from the available dictionary keys."""
        # 1. Exact match
        for k in keys:
            if str(k).lower().strip() in candidates:
                return k
        # 2. Substring match
        for k in keys:
            k_lower = str(k).lower().strip()
            if any(cand in k_lower for cand in candidates):
                return k
        return None

    def map_and_validate(self, raw_data: List[Dict[str, Any]]) -> Tuple[List[NormalizedExample], IntakeMetadata]:
        if not raw_data:
            return [], IntakeMetadata()
            
        # Discover mappings from the first non-empty row
        sample_keys = []
        for row in raw_data:
            if isinstance(row, dict) and row:
                sample_keys = list(row.keys())
                break
                
        text_key = self._find_key(sample_keys, self.TEXT_CANDIDATES)
        label_key = self._find_key(sample_keys, self.LABEL_CANDIDATES)
        
        if not text_key or not label_key:
            logger.warning(f"Could not automatically determine mapping. Found Text: {text_key}, Label: {label_key}. Fallback to exact 'text' and 'label'.")
            text_key = "text"
            label_key = "label"
        
        logger.info(f"Auto-mapping active: Mapping '{text_key}' -> 'text' and '{label_key}' -> 'label'.")

        valid_items = []
        dropped_count = 0
        distribution = {}
        
        for row in raw_data:
            if not isinstance(row, dict):
                dropped_count += 1
                continue
                
            text_val = str(row.get(text_key, "")).strip()
            label_val = str(row.get(label_key, "")).strip()
            
            # Application of Quarantine Rule: Drop missing/empty required fields
            if not text_val or not label_val:
                dropped_count += 1
                continue
                
            # Create a normalized example, absorbing extra columns
            safe_row = {k: v for k, v in row.items() if k not in {text_key, label_key}}
            safe_row["text"] = text_val
            safe_row["label"] = label_val
            
            try:
                norm_ex = NormalizedExample(**safe_row)
                valid_items.append(norm_ex)
                
                # Track distribution
                distribution[label_val] = distribution.get(label_val, 0) + 1
            except Exception as e:
                logger.warning(f"Validation failed for a row: {e}")
                dropped_count += 1

        unique_labels = sorted(list(distribution.keys()))
        
        metadata = IntakeMetadata(
            total_raw_rows=len(raw_data),
            total_valid_rows=len(valid_items),
            total_dropped_rows=dropped_count,
            unique_labels=unique_labels,
            label_distribution=distribution
        )
        
        return valid_items, metadata
