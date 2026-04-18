import os
import json
import csv
import logging
from typing import List, Dict, Any, Tuple

from .models import NormalizedExample, IntakeMetadata
from ..utils import save_jsonl

logger = logging.getLogger(__name__)

class BaseIntake:
    """
    Abstract Base Class for Intake and Validation phase.
    Handles agnostic file reading and basic saving logic.
    """
    
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.input_dir = f"runs/{run_id}/input"
        os.makedirs(self.input_dir, exist_ok=True)
        
    def read_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Parses a file into a raw list of dicts based on its extension."""
        if not os.path.exists(file_path):
            logger.error(f"Input file not found: {file_path}")
            raise FileNotFoundError(f"Missing input seed file: {file_path}")
            
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.csv':
            return self._parse_csv(file_path)
        elif ext == '.json':
            return self._parse_json(file_path)
        elif ext == '.jsonl':
            return self._parse_jsonl(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}. Only .csv, .json, and .jsonl are supported.")

    def _parse_csv(self, file_path: str) -> List[Dict[str, Any]]:
        rows = []
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            # Normalize headers immediately (lowercase, strip, replace spaces with underscores)
            reader.fieldnames = [
                (header.strip().lower().replace(" ", "_") if header else f"col_{i}")
                for i, header in enumerate(reader.fieldnames or [])
            ]
            for row in reader:
                rows.append(row)
        return rows

    def _parse_json(self, file_path: str) -> List[Dict[str, Any]]:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, list):
                logger.warning("JSON is not a list. Wrapping in a list.")
                return [data]
            return data

    def _parse_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
        rows = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def map_and_validate(self, raw_data: List[Dict[str, Any]]) -> Tuple[List[NormalizedExample], IntakeMetadata]:
        """Child classes must implement logic to map keys and quarantine invalid rows."""
        raise NotImplementedError("map_and_validate must be implemented by subclasses.")

    def run(self, input_file_path: str) -> str:
        """
        Orchestrates the read -> map -> validate -> save cycle.
        Returns the path to the saved seed_raw.jsonl file.
        """
        logger.info(f"Starting Intake Pipeline for run_id: {self.run_id}")
        
        # 1. Read
        raw_data = self.read_file(input_file_path)
        logger.info(f"Loaded {len(raw_data)} raw rows from {input_file_path}.")
        
        # 2. Map and Validate
        valid_items, metadata = self.map_and_validate(raw_data)
        
        # 3. Save Cleaned Seed Data
        seed_raw_path = os.path.join(self.input_dir, "seed_raw.jsonl")
        save_data = [item.model_dump() for item in valid_items]
        save_jsonl(save_data, seed_raw_path)
        
        # 4. Save Metadata for Node 2
        metadata_path = os.path.join(self.input_dir, "task_request.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata.model_dump(), f, ensure_ascii=False, indent=2)
            
        logger.info(f"Intake Sequence Complete. Valid rows: {metadata.total_valid_rows}/{metadata.total_raw_rows}")
        return seed_raw_path
