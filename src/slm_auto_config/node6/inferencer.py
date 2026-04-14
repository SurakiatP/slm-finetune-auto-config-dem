import torch
import logging
import json
from typing import Dict, Any, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from .parser import ResponseParser
from .models import InferenceResponse

logger = logging.getLogger(__name__)

class ClassificationInferencer:
    """
    Handles model loading and execution for classification tasks.
    Supports LoRA adapters and extracts confidence scores from token probabilities.
    """
    def __init__(self, base_model_path: str, adapter_path: Optional[str] = None):
        logger.info(f"Loading tokenizer from {base_model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        
        logger.info(f"Loading base model from {base_model_path}")
        # Load in float16 for efficiency; device_map="auto" handles GPU/CPU placement
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        if adapter_path:
            logger.info(f"Applying LoRA adapter from {adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        
        self.parser = ResponseParser()

    def predict(self, text: str, role: str, task: str, labels_list: List[str]) -> InferenceResponse:
        """
        Runs inference on the provided text using the same prompt structure as training.
        """
        labels_str = ", ".join(labels_list)
        # Identical prompt template to Node 4
        prompt_prefix = f"{role} {task} The available categories are: {labels_str}. Return the result in a valid JSON format with a single 'label' key."
        full_prompt = f"{prompt_prefix}\n\nText: {text}"
        
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.1, # Low temperature for consistent classification
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        # 1. Decode raw text output
        input_len = inputs.input_ids.shape[-1]
        generated_tokens = outputs.sequences[0][input_len:]
        raw_output = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # 2. Calculate Confidence Score
        # We use compute_transition_scores to get logprobs of the generated sequence
        try:
            # We ignore the first token if it's a padding/bos token depending on tokenizer
            transition_scores = self.model.compute_transition_scores(
                outputs.sequences, 
                outputs.scores, 
                normalize_logits=True
            )
            # transition_scores[0] contains logprobs for the single generated sequence
            logprobs = transition_scores[0]
            probs = torch.exp(logprobs)
            # Average probability across tokens in the label JSON
            avg_confidence = float(torch.mean(probs).item())
        except Exception as e:
            logger.warning(f"Could not calculate transition scores: {e}")
            avg_confidence = 0.0

        # 3. Parse JSON label
        parsed_data = self.parser.parse_classification_output(raw_output)
        
        return InferenceResponse(
            label=parsed_data.get("label", "unknown"),
            confidence=avg_confidence,
            raw_output=raw_output
        )
