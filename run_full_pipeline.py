import os
import sys
import logging
import argparse
from dotenv import load_dotenv

# Add src to Python Path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from slm_auto_config.node1 import get_intake
from slm_auto_config.node2 import get_sdg_generator
from slm_auto_config.node3 import get_splitter
from slm_auto_config.utils import setup_logging

# Load environment variables (API Keys)
load_dotenv()

def run_pipeline(input_file: str, task_type: str, run_id: str, target_count: int, task_description: str):
    # 0. Setup Global Logging
    setup_logging(run_id, "pipeline")
    logger = logging.getLogger("orchestrator")
    
    logger.info("="*50)
    logger.info(f"🚀 STARTING SLM AUTO-CONFIG PIPELINE: {run_id}")
    logger.info("="*50)

    # ---------------------------------------------------------
    # NODE 1: INTAKE & VALIDATION
    # ---------------------------------------------------------
    logger.info("\n>>> [NODE 1] Intake & Validation")
    intake = get_intake(task_type=task_type, run_id=run_id)
    seed_raw_path = intake.run(input_file)
    
    # Read metadata produced by Node 1
    import json
    metadata_path = f"runs/{run_id}/input/task_request.json"
    if not os.path.exists(metadata_path):
        logger.error(f"Node 1 failed to produce metadata at {metadata_path}")
        return

    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    logger.info(f"✅ Node 1 Complete. Valid Items: {metadata['total_valid_rows']}")

    # ---------------------------------------------------------
    # NODE 2: SYNTHETIC DATA GENERATION (SDG)
    # ---------------------------------------------------------
    logger.info("\n>>> [NODE 2] Synthetic Data Generation")
    
    generator = get_sdg_generator(
        task_type=task_type,
        task_description=task_description,
        target_count=target_count
    )
    
    api_kwargs = {
        "base_url": os.getenv("OPENROUTER_BASE_URL"),
        "api_key": os.getenv("OPENROUTER_API_KEY"),
        "model_name": "qwen/qwen3-235b-a22b-2507",
        "rule_model_name": "qwen/qwen3.6-plus",
        "judge_model_name": "openai/gpt-4o-mini"
    }
    
    synthetic_output_path = f"runs/{run_id}/synthetic/generated.json"
    
    generator.run(
        seed_data_path=seed_raw_path,
        output_path=synthetic_output_path,
        api_kwargs=api_kwargs
    )
    
    logger.info(f"✅ Node 2 Complete. Synthetic data saved to: {synthetic_output_path}")

    # ---------------------------------------------------------
    # NODE 3: SPLIT & FORMAT DATA
    # ---------------------------------------------------------
    logger.info("\n>>> [NODE 3] Split & Format Data (Oumi)")
    
    splitter = get_splitter(task_type=task_type, run_id=run_id)
    splitter.split_data(
        synthetic_path=synthetic_output_path,
        seed_path=seed_raw_path
    )
    
    logger.info(f"✅ Node 3 Complete. Split data and Oumi datasets available in: runs/{run_id}/data/")

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------
    logger.info("\n" + "="*50)
    logger.info(f"🎉 PIPELINE SUCCESS: {run_id}")
    logger.info("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SLM Auto-Config Pipeline (Nodes 1-3)")
    
    parser.add_argument("--input", type=str, default="data/raw/mock_seed.csv", help="Path to input file (csv, json, jsonl)")
    parser.add_argument("--task", type=str, default="จำแนกเอกสารและสัญญาทางกฎหมายของไทยออกเป็นหมวดหมู่", help="Description of the your task")
    parser.add_argument("--id", type=str, default="demo_run", help="Unique Run ID for this execution")
    parser.add_argument("--count", type=int, default=10, help="Target number of synthetic samples to generate")
    parser.add_argument("--type", type=str, default="classification", help="Task type (default: classification)")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ Error: Input file {args.input} not found.")
        sys.exit(1)
        
    run_pipeline(
        input_file=args.input,
        task_description=args.task,
        run_id=args.id,
        target_count=args.count,
        task_type=args.type
    )
