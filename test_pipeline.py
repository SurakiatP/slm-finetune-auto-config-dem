import os
import sys
import json

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from slm_auto_config.node3 import get_splitter
from slm_auto_config.node4 import get_config_generator, ModelParams, PeftParams, TrainingParams
from slm_auto_config.node5 import ExecutorGenerator

def main():
    run_id = "test_full_pipeline_001"
    synthetic_path = "synthetic_data.json"
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    
    print(f"Starting Full Pipeline Test: {run_id}")
    
    # ---------------------------------------------------------
    # NODE 3: SPLIT DATA
    # ---------------------------------------------------------
    print("\n--- [Node 3] Splitting Data ---")
    splitter = get_splitter(task_type="classification", run_id=run_id)
    splitter.split_data(synthetic_path=synthetic_path)
    print("Done Node 3 Complete.")

    # ---------------------------------------------------------
    # NODE 4: HYPERPARAMETER CONFIG
    # ---------------------------------------------------------
    print("\n--- [Node 4] Generating Configs ---")
    generator = get_config_generator(task_type="classification", run_id=run_id, model_name=model_name)
    
    model = ModelParams(model_name=model_name)
    peft = PeftParams()
    training = TrainingParams()
    
    # Generate Training YAML
    generator.generate_training_yaml(model, peft, training)
    # Generate Tuning YAML
    search_space = generator.get_default_search_space()
    generator.generate_tuning_yaml(model, peft, training, search_space)
    print("Done Node 4 Complete.")

    # ---------------------------------------------------------
    # NODE 5: FINE-TUNING EXECUTION
    # ---------------------------------------------------------
    print("\n--- [Node 5] Generating Execution Scripts ---")
    executor = ExecutorGenerator(run_id=run_id)
    executor.generate_scripts()
    print("Done Node 5 Complete.")

    # ---------------------------------------------------------
    # FINAL VERIFICATION
    # ---------------------------------------------------------
    print("\n--- Final Verification ---")
    expected_paths = [
        f"runs/{run_id}/data/train.jsonl",
        f"runs/{run_id}/data/data_report.json",
        f"runs/{run_id}/configs/train.yaml",
        f"runs/{run_id}/configs/tune.yaml"
    ]
    
    success = True
    for p in expected_paths:
        if os.path.exists(p):
            print(f"[FOUND] {p}")
        else:
            print(f"[MISSING] {p}")
            success = False
            
    if success:
        print("\nPipeline working effectively! All files recreated successfully.")
    else:
        print("\nPipeline failed. Some files are missing.")

    # ---------------------------------------------------------
    # PHASE 2: AUTO-BRIDGE (AFTER TUNING)
    # ---------------------------------------------------------
    print("\n--- [Phase 2] Checking for Tuning Results to Bridge ---")
    potential_csv_paths = [
        f"runs/{run_id}/training/output/trials_results.csv",
        "output/trials_results.csv",
        f"runs/{run_id}/output/trials_results.csv"
    ]
    csv_path = next((p for p in potential_csv_paths if os.path.exists(p)), None)

    if csv_path:
        from slm_auto_config.node5.auto_bridge import AutoBridge
        bridge = AutoBridge(run_id=run_id, task_type="classification", model_name=model_name)
        bridge.bridge_to_final_run(direction="minimize", final_epochs=5)
        print("Done Phase 2 Complete. Final training scripts ready.")
    else:
        print("Tuning results not found locally. Perform 'sync_from_vast.sh' first OR ensure tuning finished on this machine.")

if __name__ == "__main__":
    main()
