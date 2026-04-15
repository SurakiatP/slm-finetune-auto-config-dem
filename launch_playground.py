import os
import sys
import json
import argparse
import logging

# Ensure src directory is in the path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from slm_auto_config.node6 import get_inferencer, get_playground
from slm_auto_config.utils import setup_logging

logger = logging.getLogger(__name__)

def discover_adapter_path(run_id: str) -> str:
    """
    Search for the best available model adapter in a prioritized order.
    Returns None if no adapter is found.
    """
    # 1. Check for Final Output (Priority 1)
    final_path = f"runs/{run_id}/training/final_output"
    if os.path.exists(os.path.join(final_path, "adapter_config.json")):
        logger.info(f"🎯 Found Final Trained Model at: {final_path}")
        return final_path
    
    # 2. Check for Metadata best_model_path (Priority 2)
    metadata_path = f"runs/{run_id}/evaluation/metadata.json"
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                best_path = metadata.get("best_model_path")
                if best_path and os.path.exists(os.path.join(best_path, "adapter_config.json")):
                    logger.info(f"✅ Found best model from metadata at: {best_path}")
                    return best_path
        except Exception as e:
            logger.warning(f"Failed to parse metadata.json: {e}")

    # 3. Last resort: Look for any trial that has an adapter (Priority 3)
    trials_glob = f"runs/{run_id}/training/output/trial_*"
    import glob
    for trial_dir in glob.glob(trials_glob):
        if os.path.exists(os.path.join(trial_dir, "adapter_config.json")):
            logger.info(f"📂 Falling back to discovered trial: {trial_dir}")
            return trial_dir

    return None

def main():
    parser = argparse.ArgumentParser(description="Launch the SLM Legal Classification Playground.")
    parser.add_argument("--run_id", type=str, required=True, help="The Run ID folder to use")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="HuggingFace path for base model")
    parser.add_argument("--share", action="store_true", default=True, help="Create a public share link")
    
    args = parser.parse_args()
    
    # Correctly initialize logging with run_id
    setup_logging(args.run_id, "playground")
    logger.info(f"🚀 Launching Playground for Run: {args.run_id}")

    # Discover adapter
    adapter_path = discover_adapter_path(args.run_id)
    
    if not adapter_path:
        logger.warning("⚠️ No LoRA adapter found. Model will run without fine-tuning weights.")

    # Initialization and Launch
    try:
        logger.info(f"⚡ Initializing Inferencer (Base: {args.base_model}, Adapter: {adapter_path})...")
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
