import json
import os
import sys
import logging
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
# Add src to path to import other nodes
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from slm_auto_config.node5.analyser import MetricsAnalyser
from slm_auto_config.node5.visualizer import Visualizer
from slm_auto_config.node5.models import EvaluationMetrics, LabelMetric

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_model_response(response_str: str) -> str:
    """
    Tries to extract the 'label' from the model's response string.
    Handles raw JSON or plain text matches.
    """
    response_str = response_str.strip()
    # Try parsing as JSON first
    try:
        # Simple cleanup for common hallucination patterns
        if "```json" in response_str:
            response_str = response_str.split("```json")[1].split("```")[0].strip()
        elif "{" in response_str:
            response_str = response_str[response_str.find("{"):response_str.rfind("}")+1]
            
        data = json.loads(response_str)
        if isinstance(data, dict) and "label" in data:
            return str(data["label"]).strip()
    except:
        pass
    
    # Fallback: line based extraction
    for line in response_str.split("\n"):
        if '"label":' in line:
            return line.split('"label":')[-1].replace('"', '').replace(',', '').strip()
            
    return "unknown"

def run_evaluation(run_id: str):
    logger.info(f"📊 Starting Node 7 Evaluation for run: {run_id}")
    
    test_path = f"runs/{run_id}/data/test.jsonl"
    preds_path = f"runs/{run_id}/evaluation/predictions.jsonl"
    eval_dir = f"runs/{run_id}/evaluation"
    
    if not os.path.exists(test_path) or not os.path.exists(preds_path):
        logger.error(f"Missing files: test={os.path.exists(test_path)}, preds={os.path.exists(preds_path)}")
        return
    
    # 1. Load Ground Truth
    ground_truth = []
    with open(test_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            # Find the label in the messages
            # For SFT format: messages: [{"role": "user", ...}, {"role": "assistant", "content": "{\"label\": \"...\"}"}]
            for msg in data.get("messages", []):
                if msg["role"] == "assistant":
                    content = msg["content"]
                    try:
                        parsed = json.loads(content)
                        ground_truth.append(parsed.get("label", "unknown"))
                    except:
                        ground_truth.append("unknown")
    
    # 2. Load Predictions
    predictions = []
    with open(preds_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            raw_response = data.get("response", data.get("generated_text", ""))
            predictions.append(parse_model_response(raw_response))
            
    # Match lengths
    if len(ground_truth) != len(predictions):
        logger.warning(f"Length mismatch: GT={len(ground_truth)}, Preds={len(predictions)}. Truncating to minimum.")
        min_len = min(len(ground_truth), len(predictions))
        ground_truth = ground_truth[:min_len]
        predictions = predictions[:min_len]
        
    # 3. Calculate Metrics
    labels = sorted(list(set(ground_truth)))
    report = classification_report(ground_truth, predictions, output_dict=True, zero_division=0)
    cm = confusion_matrix(ground_truth, predictions, labels=labels).tolist()
    
    # 4. Standardize to Node 5 Models
    label_metrics = {}
    for label in labels:
        if label in report:
            label_metrics[label] = LabelMetric(
                precision=report[label]['precision'],
                recall=report[label]['recall'],
                f1=report[label]['f1-score'],
                support=int(report[label]['support'])
            )
            
    metrics = EvaluationMetrics(
        accuracy=report['accuracy'],
        macro_f1=report['macro avg']['f1-score'],
        label_metrics=label_metrics,
        confusion_matrix=cm,
        labels=labels
    )
    
    # 5. Save and Visualize
    metrics_path = f"{eval_dir}/metrics.json"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics.model_dump(), f, indent=4, ensure_ascii=False)
        
    logger.info(f"✅ Metrics saved to {metrics_path}")
    
    viz = Visualizer(run_id)
    pdf_path = viz.generate_pdf_report(metrics)
    logger.info(f"📄 PDF Report generated at {pdf_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_eval.py <run_id>")
        sys.exit(1)
    run_evaluation(sys.argv[1])
