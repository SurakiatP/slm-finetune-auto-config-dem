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

def clean_label(label: str) -> str:
    """Standardizes label naming to match canonical categories."""
    if not label: 
        return "unknown"
    
    label = label.strip().lower()
    
    # Mapping logic for legal categories
    if "nda" in label or "ความลับ" in label:
        return "ข้อตกลงรักษาความลับ (NDA)"
    if "จ้างงาน" in label:
        return "สัญญาจ้างงาน"
    if "เช่า" in label:
        return "สัญญาเช่า"
    if "จัดซื้อ" in label or "จัดจ้าง" in label:
        return "เอกสารจัดซื้อจัดจ้าง"
        
    return "unknown"

def parse_model_response(response_str: str) -> str:
    """
    Tries to extract the 'label' from the model's response string.
    Handles raw JSON or plain text matches.
    """
    response_str = response_str.strip()
    extracted = "unknown"
    
    # Try parsing as JSON first
    try:
        if "```json" in response_str:
            response_str = response_str.split("```json")[1].split("```")[0].strip()
        elif "{" in response_str:
            response_str = response_str[response_str.find("{"):response_str.rfind("}")+1]
            
        data = json.loads(response_str)
        if isinstance(data, dict) and "label" in data:
            extracted = str(data["label"])
    except:
        # Fallback: line based extraction
        for line in response_str.split("\n"):
            if '"label":' in line:
                extracted = line.split('"label":')[-1].replace('"', '').replace(',', '').strip()
                break
            
    return clean_label(extracted)

def run_evaluation(run_id: str):
    logger.info(f"📊 Starting Node 7 Evaluation for run: {run_id}")
    
    preds_path = f"runs/{run_id}/evaluation/predictions.jsonl"
    eval_dir = f"runs/{run_id}/evaluation"
    
    if not os.path.exists(preds_path):
        logger.error(f"Missing file: preds={os.path.exists(preds_path)}")
        return
    
    ground_truth = []
    predictions = []
    
    # Oumi 0.7 Inference output is a JSONL of Conversations
    with open(preds_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                msgs = data.get("messages", [])
                
                # In Oumi Inference on a labeled dataset:
                # msgs[0] = User Prompt
                # msgs[1] = Ground Truth (Assistant)
                # msgs[2] = Model Prediction (Assistant)
                
                # Extract Ground Truth (usually index 1)
                gt_val = "unknown"
                for msg in msgs[1:-1]: # Look in between prompt and last response
                    if msg["role"] == "assistant":
                        gt_val = parse_model_response(msg["content"])
                        break
                ground_truth.append(gt_val)
                
                # Extract Prediction (the very last message)
                if len(msgs) > 0 and msgs[-1]["role"] == "assistant":
                    pred_val = parse_model_response(msgs[-1]["content"])
                    predictions.append(pred_val)
                else:
                    predictions.append("unknown")
                    
            except Exception as e:
                logger.warning(f"Failed to parse line: {e}")
        
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
