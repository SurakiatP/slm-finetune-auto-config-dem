import os
import sys
import json
import argparse
import logging

# Ensure src directory is in the path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from slm_auto_config.node7 import get_exporter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Export the fine-tuned SLM to multiple formats.")
    parser.add_argument(
        "--run_id", 
        type=str, 
        required=True, 
        help="Run ID folder to export (e.g., test_full_pipeline_001)"
    )
    parser.add_argument(
        "--base_model", 
        type=str, 
        default="Qwen/Qwen2.5-0.5B-Instruct", 
        help="HuggingFace path for the original base model."
    )
    parser.add_argument(
        "--formats", 
        nargs="+", 
        default=["safetensors", "gguf", "onnx"], 
        help="Desired export formats: safetensors, gguf, onnx"
    )
    
    args = parser.parse_args()

    # Discover the best adapter discovered by Node 5
    metadata_path = f"runs/{args.run_id}/evaluation/metadata.json"
    adapter_path = None
    
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
                adapter_path = meta.get("best_model_path")
                if adapter_path:
                    logger.info(f"✅ Found best model metadata for export: {adapter_path}")
        except Exception as e:
            logger.error(f"❌ Failed to parse metadata.json: {e}")
    else:
        logger.warning(f"⚠️ No best model metadata found at {metadata_path}. Will export base model only (if it exists).")

    try:
        # 1. Initialize Exporter via Factory
        exporter = get_exporter(
            task_type="classification", 
            run_id=args.run_id, 
            base_model=args.base_model, 
            adapter_path=adapter_path
        )
        
        # 2. Run the Export
        logger.info(f"🚀 Starting multi-format export process for {args.run_id}...")
        results = exporter.export(formats=args.formats)
        
        # 3. Final Summary
        logger.info("✨ All requested formats have been processed!")
        print("\n" + "="*50)
        print(f"EXPORT SUMMARY - RUN: {args.run_id}")
        print("="*50)
        for fmt, path in results.items():
            print(f"  - [{fmt.upper()}]: {path}")
        print("="*50)
        print(f"Documentation and Metadata saved in runs/{args.run_id}/export/")
            
    except Exception as e:
        logger.critical(f"💥 Export failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()
