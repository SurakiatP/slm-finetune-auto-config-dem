import torch
import logging
import json
import os
import gradio as gr
from typing import List, Optional, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from .base import BaseInferencer, BasePlayground
from .models import InferenceResponse
from .parser import ResponseParser

logger = logging.getLogger(__name__)

class ClassificationInferencer(BaseInferencer):
    """
    Concrete Inferencer for Classification tasks.
    """
    def __init__(self, base_model_path: str, adapter_path: Optional[str] = None):
        super().__init__(base_model_path, adapter_path)
        logger.info(f"Loading ClassificationInferencer: {base_model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        if adapter_path:
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.parser = ResponseParser()

    def _clean_label(self, label: str, valid_labels: List[str]) -> str:
        """Standardizes label naming dynamically using fuzzy match against valid_labels."""
        if not label: return "unknown"
        label = label.strip().lower()
        
        # 1. Exact or strict lower match
        for v in valid_labels:
            if label == v.lower():
                return v
                
        # 2. Substring match
        for v in valid_labels:
            if label in v.lower() or v.lower() in label:
                return v
                
        return "unknown"

    def predict(self, text: str, **kwargs) -> InferenceResponse:
        """
        Runs classification inference using the EXACT prompt format from training.
        """
        role = kwargs.get("role", "You are an expert document classifier.")
        task = kwargs.get("task", "Your task is to identify the most appropriate category for the provided text.")
        labels_list = kwargs.get("labels_list", [])
        
        labels_str = ", ".join(labels_list)
        
        # 1. Apply Chat Template using separated System and User messages
        instructions = (
            f"{task} The available categories are: {labels_str}. "
            f"Return the result in a valid JSON format with a single 'label' key."
        )
        
        messages = [
            {"role": "system", "content": role},
            {"role": "user", "content": f"{instructions}\n\nText: {text}"}
        ]
        
        full_prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
        
        # 2. Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.1,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        # Decode raw text
        input_len = inputs.input_ids.shape[-1]
        generated_tokens = outputs.sequences[0][input_len:]
        raw_output = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # 3. Calculate Confidence
        try:
            transition_scores = self.model.compute_transition_scores(
                outputs.sequences, 
                outputs.scores, 
                normalize_logits=True
            )
            probs = torch.exp(transition_scores[0])
            avg_confidence = float(torch.mean(probs).item())
        except Exception as e:
            logger.warning(f"Confidence score calculation failed: {e}")
            avg_confidence = 0.0

        # 4. Parse & Normalize
        parsed_data = self.parser.parse_classification_output(raw_output)
        predicted_label = parsed_data.get("label", "unknown")
        
        # Apply normalization (mapping "NDA" -> "ข้อตกลงรักษาความลับ (NDA)")
        normalized_label = self._clean_label(predicted_label, labels_list)
        
        return InferenceResponse(
            label=normalized_label,
            confidence=avg_confidence,
            raw_output=raw_output
        )

class ClassificationPlayground(BasePlayground):
    """
    Concrete Playground for Classification tasks.
    """
    def _load_run_context(self):
        report_path = f"runs/{self.run_id}/data/data_report.json"
        role = "You are an expert document classifier."
        task = "Your task is to identify the most appropriate category for the provided text."
        labels = []
        if os.path.exists(report_path):
            try:
                with open(report_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    labels = data.get("labels", [])
            except Exception as e:
                logger.warning(f"Failed to load labels from report: {e}")
        return role, task, labels

    def launch(self, share: bool = True):
        role, task, labels = self._load_run_context()
        
        def classify_text(text: str):
            if not text or not text.strip():
                return "No input provided.", 0.0, ""
            
            response = self.inferencer.predict(
                text=text,
                role=role,
                task=task,
                labels_list=labels
            )
            return response.label, response.confidence, response.raw_output

        with gr.Blocks(title=f"Classification Playground - {self.run_id}", theme=gr.themes.Soft()) as demo:
            gr.Markdown(f"# ⚖️ SLM Legal Classification Playground ({self.run_id})")
            gr.Markdown("Interactive inference for category prediction.")
            
            with gr.Row():
                with gr.Column(scale=2):
                    input_text = gr.Textbox(label="Document Content", lines=12, placeholder="Paste Thai legal text...")
                    submit_btn = gr.Button("Classify", variant="primary")
                
                with gr.Column(scale=1):
                    output_label = gr.Label(label="Prediction Result")
                    output_conf = gr.Number(label="Confidence Score", precision=4)
                    with gr.Accordion("Raw Model JSON Response", open=False):
                        output_raw = gr.Code(label="JSON", language="json")

            submit_btn.click(
                fn=classify_text,
                inputs=input_text,
                outputs=[output_label, output_conf, output_raw]
            )

            gr.Markdown("---")
            gr.Markdown("**Backend:** Strategy: Classification | Device: CUDA/Auto")

        demo.launch(share=share, server_name="0.0.0.0")
