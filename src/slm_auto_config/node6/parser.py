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

        # 2. Try to find a JSON block using regex (more aggressive)
        json_match = re.search(r'\{(?:[^{}]|(?R))*\}', text, re.DOTALL) # Handle nested {}
        if not json_match:
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            
        if json_match:
            try:
                # Clean up problematic characters inside the match
                clean_json = json_match.group().replace('\n', ' ').strip()
                return json.loads(clean_json)
            except json.JSONDecodeError:
                pass
        
        # 3. Try common key-value extraction (more flexible)
        # Matches "label": "value" or label: value or 'label': 'value'
        label_match = re.search(r'["\']?label["\']?\s*[:=]\s*["\']?([^"\'\s,{}]+)["\']?', text, re.IGNORECASE)
        if label_match:
            return {"label": label_match.group(1)}
            
        # 4. If nothing else, try to see if the whole response is just a single word (the label)
        if len(text.split()) < 5 and not '{' in text:
            return {"label": text}

        # If all fails, return a special error label
        return {
            "label": "ERROR: Parsing Failed",
            "raw": text
        }
