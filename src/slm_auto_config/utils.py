import json
import os
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

def load_json(path: str) -> List[dict]:
    """Loads a JSON file if it exists."""
    if not path or not os.path.exists(path):
        logger.warning(f"File not found: {path}")
        return []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading {path}: {e}")
        return []

def save_jsonl(data: List[dict], path: str):
    """Saves a list of dicts to a JSONL file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    logger.info(f"Saved {len(data)} items to {path}")
