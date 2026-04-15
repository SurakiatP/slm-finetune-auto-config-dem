import os
import sys
import json
import argparse
import logging

# Ensure src directory is in the path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from slm_auto_config.node6 import get_inferencer, get_playground
from slm_auto_config.utils import setup_logging

# Initialize Logging
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Launch the SLM Legal Classification Playground.")
    parser.add_argument(
        "--run_id", 
        type=str, 
        required=True, 
        help="The Run ID folder to use (e.g., test_full_pipeline_001)"
    )
    parser.add_argument(
        "--base_model", 
        type=str, 
        default="Qwen/Qwen2.5-0.5B-Instruct", 
        help="HuggingFace path for the base model."
    )
    parser.add_argument(
        "--share", 
        action="store_true", 
        default=True,
        help="Create a publicly accessible share link (gradio.live)."
    )
    
    args = parser.parse_args()
    
    # Path to find the best adapter discovered by Node 5
    metadata_path = f"runs/{args.run_id}/evaluation/metadata.json"
    adapter_path = None
    
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                adapter_path = metadata.get("best_model_path")
                if adapter_path:
                    logger.info(f"✅ Found best model metadata. Loading adapter from: {adapter_path}")
        except Exception as e:
            logger.error(f"❌ Failed to parse metadata.json: {e}")
    else:
        logger.warning(f"⚠️ No metadata.json found at {metadata_path}. Running with base model only.")

    # Initialization and Launch
    try:
        logger.info("⚡ Initializing Inferencer (Strategy: Classification)...")
        # Currently defaults to classification, expandable to other types via CLI/Metadata
        inferencer = get_inferencer(
            task_type="classification",
            base_model_path=args.base_model,
            adapter_path=adapter_path
        )
        
        logger.info("🎨 Starting Gradio UI...")
        ui = get_playground(
            task_type="classification",
            inferencer=inferencer, 
            run_id=args.run_id
        )
        ui.launch(share=args.share)
        
    except Exception as e:
        logger.critical(f"💥 Critical error during launch: {e}", exc_info=True)

if __name__ == "__main__":
    main()
