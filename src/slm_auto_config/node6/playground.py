import gradio as gr
import logging
import os
import json
from typing import List, Optional
from .inferencer import ClassificationInferencer
from .models import InferenceResponse

logger = logging.getLogger(__name__)

class PlaygroundUI:
    """
    Gradio-based interactive UI for testing the fine-tuned model.
    """
    def __init__(self, inferencer: ClassificationInferencer, run_id: str):
        self.inferencer = inferencer
        self.run_id = run_id
        
        # Load run metadata to get the original role, task, and labels
        self.role, self.task, self.labels = self._load_run_context()

    def _load_run_context(self):
        """Attempts to recover labels and instructions from Node 3/4 artifacts."""
        report_path = f"runs/{self.run_id}/data/data_report.json"
        # Default fallbacks
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

    def classify_text(self, text: str):
        """Wrapper for the Gradio interface."""
        if not text or not text.strip():
            return "No input provided.", 0.0, ""
        
        try:
            response = self.inferencer.predict(
                text=text,
                role=self.role,
                task=self.task,
                labels_list=self.labels
            )
            return response.label, response.confidence, response.raw_output
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return f"Error: {str(e)}", 0.0, ""

    def launch(self, share: bool = True):
        """Starts the Gradio web server."""
        with gr.Blocks(title=f"Playground - {self.run_id}", theme=gr.themes.Soft()) as demo:
            gr.Markdown(f"# ⚖️ SLM Legal Classification Playground ({self.run_id})")
            gr.Markdown("Paste your Thai legal text below to see how the model categorizes it.")
            
            with gr.Row():
                with gr.Column(scale=2):
                    input_text = gr.Textbox(
                        label="Document Content", 
                        placeholder="กรุณาวางเนื้อหาเอกสารภาษาไทยที่นี่...", 
                        lines=12
                    )
                    with gr.Row():
                        clear_btn = gr.Button("Clear")
                        submit_btn = gr.Button("Classify", variant="primary")
                
                with gr.Column(scale=1):
                    output_label = gr.Label(label="Prediction Result")
                    output_conf = gr.Number(label="Confidence Score (0.0 - 1.0)", precision=4)
                    with gr.Accordion("Raw Model JSON Response", open=False):
                        output_raw = gr.Code(label="JSON", language="json")

            # Wiring
            submit_btn.click(
                fn=self.classify_text,
                inputs=input_text,
                outputs=[output_label, output_conf, output_raw]
            )
            clear_btn.click(lambda: ["", None, ""], None, [input_text, output_label, output_raw])

            gr.Markdown("---")
            gr.Markdown("**System Info:** Running on Vast.ai | Model: Qwen2.5 | PEFT: Enabled")

        # Launch server
        demo.launch(share=share, server_name="0.0.0.0")
