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

def setup_logging(run_id: str, log_name: str):
    """
    Sets up logging to both console and a file in the run's log directory.
    """
    log_dir = f"runs/{run_id}/logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = f"{log_dir}/{log_name}.log"

    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging initialized. Log file: {log_file}")
