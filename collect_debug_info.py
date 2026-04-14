import os
import zipfile
import subprocess
import argparse
from datetime import datetime

def collect_debug_info(run_id: str):
    """
    Collects all logs, configs, and system information into a single ZIP file
    to help Antigravity debug issues on Vast.ai.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_name = f"debug_report_{run_id}_{timestamp}"
    zip_path = f"runs/{run_id}/{report_name}.zip"
    
    run_path = f"runs/{run_id}"
    log_dir = os.path.join(run_path, "logs")
    config_dir = os.path.join(run_path, "configs")
    eval_dir = os.path.join(run_path, "evaluation")
    
    if not os.path.exists(run_path):
        print(f"❌ Error: Run directory {run_path} not found.")
        return

    print(f"📦 Gathering debug information for run: {run_id}...")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # 1. Capture Logs
        if os.path.exists(log_dir):
            print("  - Adding logs...")
            for root, _, files in os.walk(log_dir):
                for file in files:
                    full_path = os.path.join(root, file)
                    arcname = os.path.relpath(full_path, run_path)
                    zipf.write(full_path, arcname=arcname)
        
        # 2. Capture Configs (Yaml files)
        if os.path.exists(config_dir):
            print("  - Adding configurations...")
            for file in os.listdir(config_dir):
                if file.endswith(('.yaml', '.yml', '.json')):
                    zipf.write(os.path.join(config_dir, file), arcname=f"configs/{file}")

        # 3. Capture Evaluation Metadata (if any)
        if os.path.exists(eval_dir):
            print("  - Adding evaluation metadata...")
            for file in os.listdir(eval_dir):
                if file.endswith('.json'):
                    zipf.write(os.path.join(eval_dir, file), arcname=f"evaluation/{file}")

        # 4. Generate System Diagnostics
        print("  - Gathering system diagnostics...")
        sys_info = []
        sys_info.append(f"Report Generated: {datetime.now().isoformat()}")
        sys_info.append(f"Run ID: {run_id}")
        sys_info.append("\n" + "="*40)
        sys_info.append("GPU INFO (nvidia-smi)")
        sys_info.append("="*40)
        try:
            nvidia_output = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT).decode()
            sys_info.append(nvidia_output)
        except Exception:
            sys_info.append("nvidia-smi command not found or failed.")

        sys_info.append("\n" + "="*40)
        sys_info.append("DISK SPACE (df -h)")
        sys_info.append("="*40)
        try:
            df_output = subprocess.check_output(["df", "-h", "."], stderr=subprocess.STDOUT).decode()
            sys_info.append(df_output)
        except Exception:
            sys_info.append("df command failed.")

        sys_info.append("\n" + "="*40)
        sys_info.append("INSTALLED PACKAGES (pip list)")
        sys_info.append("="*40)
        try:
            pip_output = subprocess.check_output(["pip", "list"], stderr=subprocess.STDOUT).decode()
            sys_info.append(pip_output)
        except Exception:
            sys_info.append("pip list command failed.")

        zipf.writestr("system_diagnostics.txt", "\n".join(sys_info))

    print(f"\n✨ Debug report created successfully!")
    print(f"📍 Location: {zip_path}")
    print("\nNext Steps:")
    print(f"1. Run 'bash runs/{run_id}/scripts/sync_from_vast.sh ...' on your local computer to download it.")
    print(f"2. Send the content of the logs or share the report details with Antigravity.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect debug information from a Vast.ai run.")
    parser.add_argument("--run_id", type=str, required=True, help="The ID of the run to debug.")
    args = parser.parse_args()
    collect_debug_info(args.run_id)
