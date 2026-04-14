import json
import logging
from typing import List, Dict, Optional
from sklearn.model_selection import train_test_split
from .base import BaseSplitter
from .models import OumiExample, OumiMessage

logger = logging.getLogger(__name__)

class ClassificationSplitter(BaseSplitter):
    def __init__(self, run_id: str, role: str = None, task: str = None):
        """
        Classification-specific splitter implementation.
        """
        super().__init__(run_id, "classification")
        self.role = role or "You are an expert document classifier."
        self.task = task or "Your task is to identify the most appropriate category for the provided text."
        # Fixed English template structure with dynamic label injection
        self.instruction_template = "{role} {task} The available categories are: {labels}. Return the result in a valid JSON format with a single 'label' key."

    def detect_labels(self, data: List[dict]) -> List[str]:
        labels = set()
        for item in data:
            if isinstance(item, dict) and 'label' in item and item['label']:
                labels.add(item['label'])
        return sorted(list(labels))

    def convert_to_oumi(self, data: List[dict], labels: List[str]) -> List[dict]:
        formatted_labels = ", ".join(labels)
        prompt_prefix = self.instruction_template.format(
            role=self.role,
            task=self.task,
            labels=formatted_labels
        )
        
        oumi_data = []
        for item in data:
            messages = [
                OumiMessage(role="user", content=f"{prompt_prefix}\n\nText: {item['text']}"),
                OumiMessage(role="assistant", content=json.dumps({"label": item['label']}, ensure_ascii=False))
            ]
            example = OumiExample(messages=messages)
            oumi_data.append(example.model_dump())
        return oumi_data

    def get_distribution(self, data: List[dict]) -> Dict[str, int]:
        dist = {}
        for item in data:
            label = item.get('label', 'unknown')
            dist[label] = dist.get(label, 0) + 1
        return dist

    def execute_split(self, synthetic_raw: List[dict], seed_raw: List[dict], val_ratio: float, test_ratio: float):
        train_raw = []
        val_raw = []
        test_raw = []

        if seed_raw:
            # priority for seed data in val/test
            if len(seed_raw) >= 2:
                v_raw, t_raw = train_test_split(
                    seed_raw, 
                    test_size=0.5, 
                    stratify=[s['label'] for s in seed_raw] if self._is_stratifiable(seed_raw) else None
                )
                val_raw.extend(v_raw)
                test_raw.extend(t_raw)
            else:
                val_raw.extend(seed_raw)
            # All synthetic to train
            train_raw.extend(synthetic_raw)
        else:
            # synthetic only
            if self._is_stratifiable(synthetic_raw):
                train_raw, temp_raw = train_test_split(
                    synthetic_raw, 
                    test_size=(val_ratio + test_ratio), 
                    stratify=[s['label'] for s in synthetic_raw]
                )
                val_raw, test_raw = train_test_split(
                    temp_raw, 
                    test_size=test_ratio / (val_ratio + test_ratio),
                    stratify=[s['label'] for s in temp_raw] if self._is_stratifiable(temp_raw) else None
                )
            else:
                logger.warning("Data not stratifiable, falling back to simple split.")
                train_raw, temp_raw = train_test_split(synthetic_raw, test_size=(val_ratio + test_ratio))
                val_raw, test_raw = train_test_split(temp_raw, test_size=0.5)
        
        return train_raw, val_raw, test_raw

    def _is_stratifiable(self, data: List[dict]) -> bool:
        if not data: return False
        counts = self.get_distribution(data)
        return all(count >= 2 for count in counts.values())
