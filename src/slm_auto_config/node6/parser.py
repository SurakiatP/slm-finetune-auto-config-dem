import json
import re
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class ResponseParser:
    """
    Handles robust extraction of structured JSON from LLM text responses.
    """
    def parse_classification_output(self, raw_text: str) -> Dict[str, Any]:
        """
        Attempts to parse a label from the raw model output.
        Expects: {"label": "..."}
        """
        text = raw_text.strip()
        
        # 1. Try direct JSON parsing
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 2. Try to find a JSON block using regex
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # 3. Last resort: Try common key-value extraction
        label_match = re.search(r'"label":\s*"([^"]+)"', text)
        if label_match:
            return {"label": label_match.group(1)}
            
        # If all fails, return a special error label
        return {
            "label": "ERROR: Parsing Failed",
            "raw": text
        }
