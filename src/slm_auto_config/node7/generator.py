import os
import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ModelCardGenerator:
    """
    Generates a README.md (Model Card) describing the fine-tuning 
    process, metrics, and how to use the model.
    """
    def __init__(self, run_id: str):
        self.run_id = run_id

    def generate(self, export_dir: str, export_results: Dict[str, str]):
        # Gather info from previous nodes
        metrics = self._load_json(f"runs/{self.run_id}/evaluation/metrics.json")
        data_report = self._load_json(f"runs/{self.run_id}/data/data_report.json")
        
        labels = data_report.get("labels", [])
        
        content = f"""# Fine-Tuned SLM: {self.run_id}

This model is a fine-tuned version of `Qwen2.5-0.5B-Instruct` optimized for **Thai Legal Document Classification**. 
It was created using the **SLM Auto Config** pipeline.

## 📊 Performance Metrics
Results obtained on the held-out test set (10% of total data):

- **Accuracy:** {metrics.get('accuracy', 'N/A'):.4f if isinstance(metrics.get('accuracy'), float) else 'N/A'}
- **Macro F1-Score:** {metrics.get('macro_f1', 'N/A'):.4f if isinstance(metrics.get('macro_f1'), float) else 'N/A'}

### Classification Labels
The model is trained to recognize the following categories:
{chr(10).join([f"- {label}" for label in labels])}

## 📁 Exported Formats
This package contains the model in multiple formats:
- **Hugging Face / SafeTensors**: Located in `./safetensors/`
- **GGUF (Quantization-ready)**: Located in `./gguf/`
- **ONNX (Cross-platform)**: Located in `./onnx/`

## 🚀 How to Use

### Using Transformers
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./safetensors")
tokenizer = AutoTokenizer.from_pretrained("./safetensors")
```

### Using Ollama (GGUF)
1. Convert the model to a specific quantization if needed using `llama-quantize`.
2. Create a `Modelfile`.
3. Run `ollama create my-model -f Modelfile`.

## 🛠️ Training Details
- **Framework:** Oumi AI / PEFT
- **Method:** LoRA (Merged into final weights)
- **Dataset:** Stratified Split (80/10/10)
"""
        save_path = f"{export_dir}/README.md"
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"Model card saved to {save_path}")

    def _load_json(self, path: str) -> Dict[str, Any]:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

class MetadataGenerator:
    """
    Generates a structured run_card.json for audit and reproducibility.
    """
    def __init__(self, run_id: str):
        self.run_id = run_id

    def generate(self, export_dir: str, export_results: Dict[str, str]):
        # Collect all relevant run data
        run_card = {
            "run_id": self.run_id,
            "timestamp": "2026-04-14", # Placeholder for real time
            "task_type": "classification",
            "model_architecture": "Qwen2 (Llama-like)",
            "export_formats": export_results,
            "pipeline_summary": {
                "node3_data": f"runs/{self.run_id}/data/data_report.json",
                "node5_metrics": f"runs/{self.run_id}/evaluation/metrics.json"
            }
        }
        
        save_path = f"{export_dir}/run_card.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(run_card, f, indent=4)
        logger.info(f"Run card saved to {save_path}")
