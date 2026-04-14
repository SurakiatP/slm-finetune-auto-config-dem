import os
import logging
import json
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from ..utils import load_json, save_jsonl
from .models import OumiExample, OumiMessage, DataSplitReport

logger = logging.getLogger(__name__)

class BaseSplitter(ABC):
    def __init__(self, run_id: str, task_type: str):
        """
        Base class for task-specific splitters.
        """
        self.run_id = run_id
        self.task_type = task_type

    @abstractmethod
    def detect_labels(self, data: List[dict]) -> List[str]:
        """Detect unique labels or entity types from data."""
        pass

    @abstractmethod
    def convert_to_oumi(self, data: List[dict], labels: List[str]) -> List[dict]:
        """Convert list of examples into Oumi text_sft format."""
        pass

    @abstractmethod
    def get_distribution(self, data: List[dict]) -> Dict[str, int]:
        """Calculate distribution of labels/entities."""
        pass

    @abstractmethod
    def execute_split(self, synthetic_raw: List[dict], seed_raw: List[dict], val_ratio: float, test_ratio: float):
        """Perform the actual split logic (Stratified, Random, etc.)."""
        pass

    def split_data(self, synthetic_path: str, seed_path: Optional[str] = None, val_ratio: float = 0.1, test_ratio: float = 0.1):
        """Universal split orchestration."""
        synthetic_raw = load_json(synthetic_path)
        seed_raw = load_json(seed_path) if seed_path else []
        
        if not synthetic_raw and not seed_raw:
            logger.error(f"[{self.task_type}] No data found to split.")
            return

        all_data = synthetic_raw + seed_raw
        all_labels = self.detect_labels(all_data)
        logger.info(f"[{self.task_type}] Detected labels: {all_labels}")

        train_raw, val_raw, test_raw = self.execute_split(synthetic_raw, seed_raw, val_ratio, test_ratio)

        # Convert to Oumi format
        train_oumi = self.convert_to_oumi(train_raw, all_labels)
        val_oumi = self.convert_to_oumi(val_raw, all_labels)
        test_oumi = self.convert_to_oumi(test_raw, all_labels)

        # Save files
        base_dir = f"runs/{self.run_id}/data"
        save_jsonl(train_oumi, f"{base_dir}/train.jsonl")
        save_jsonl(val_oumi, f"{base_dir}/validation.jsonl")
        save_jsonl(test_oumi, f"{base_dir}/test.jsonl")

        # Create and save report
        report = DataSplitReport(
            run_id=self.run_id,
            task_type=self.task_type,
            total_count=len(all_data),
            train_count=len(train_raw),
            val_count=len(val_raw),
            test_count=len(test_raw),
            labels=all_labels,
            label_distribution={
                "train": self.get_distribution(train_raw),
                "validation": self.get_distribution(val_raw),
                "test": self.get_distribution(test_raw)
            }
        )
        
        report_path = f"{base_dir}/data_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report.model_dump(), f, ensure_ascii=False, indent=2)
        logger.info(f"[{self.task_type}] Data report saved to {report_path}")
