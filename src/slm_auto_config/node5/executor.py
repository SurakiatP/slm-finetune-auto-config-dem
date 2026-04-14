import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class ExecutorGenerator:
    """
    Generates shell scripts for executing Oumi commands on Vast.ai and syncing files.
    """
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.script_dir = f"runs/{run_id}/scripts"
        self.log_dir = f"runs/{run_id}/logs"
        os.makedirs(self.script_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    def generate_scripts(self, remote_ip: str = "HOST_IP", remote_port: str = "PORT"):
        """
        Generates a suite of shell scripts for the run.
        """
        # 1. run_train.sh (Manual Training)
        train_sh = [
            "#!/bin/bash",
            f"echo '🚀 Starting Manual Training for {self.run_id}'",
            f"oumi train -c runs/{self.run_id}/configs/train.yaml 2>&1 | tee runs/{self.run_id}/logs/train.log",
            "echo '✅ Training Complete. Log saved to runs/{self.run_id}/logs/train.log'"
        ]
        self._write_script("run_train.sh", train_sh)

        # 2. run_tune.sh (Auto Tuning)
        tune_sh = [
            "#!/bin/bash",
            f"echo '🚀 Starting Auto Tuning for {self.run_id}'",
            f"oumi tune -c runs/{self.run_id}/configs/tune.yaml 2>&1 | tee runs/{self.run_id}/logs/tune.log",
            "echo '✅ Tuning Complete. Log saved to runs/{self.run_id}/logs/tune.log'"
        ]
        self._write_script("run_tune.sh", tune_sh)

        # 3. run_eval.sh (Standalone Evaluation on Test Set)
        eval_sh = [
            "#!/bin/bash",
            f"echo '🔍 Starting Evaluation for {self.run_id} on Test Set'",
            f"oumi evaluate -c runs/{self.run_id}/configs/train.yaml --dataset_path runs/{self.run_id}/data/test.jsonl 2>&1 | tee runs/{self.run_id}/logs/eval.log",
            "echo '✅ Evaluation Complete. Log saved to runs/{self.run_id}/logs/eval.log'"
        ]
        self._write_script("run_eval.sh", eval_sh)

        # 4. sync_to_vast.sh (Helper for Local -> Remote)
        sync_to = [
            "#!/bin/bash",
            f"echo '📤 Syncing {self.run_id} TO Vast.ai...'",
            "# Usage: ./sync_to_vast.sh [REMOTE_USER_HOST] [PORT]",
            "REMOTE=$1",
            "PORT=$2",
            "if [ -z \"$REMOTE\" ] || [ -z \"$PORT\" ]; then echo 'Usage: ./sync_to_vast.sh user@ip port'; exit 1; fi",
            f"rsync -avzP -e \"ssh -p $PORT\" runs/{self.run_id}/ $REMOTE:~/slm-auto-config/runs/{self.run_id}/",
            "echo '✅ Sync Complete.'"
        ]
        self._write_script("sync_to_vast.sh", sync_to)
        
        # 5. sync_from_vast.sh (Helper for Remote -> Local)
        sync_from = [
            "#!/bin/bash",
            f"echo '📥 Syncing {self.run_id} FROM Vast.ai...'",
            "# Usage: ./sync_from_vast.sh [REMOTE_USER_HOST] [PORT]",
            "REMOTE=$1",
            "PORT=$2",
            "if [ -z \"$REMOTE\" ] || [ -z \"$PORT\" ]; then echo 'Usage: ./sync_from_vast.sh user@ip port'; exit 1; fi",
            f"rsync -avzP -e \"ssh -p $PORT\" $REMOTE:~/slm-auto-config/runs/{self.run_id}/ runs/{self.run_id}/",
            "echo '✅ Sync Complete.'"
        ]
        self._write_script("sync_from_vast.sh", sync_from)

    def _write_script(self, name: str, lines: list):
        path = f"{self.script_dir}/{name}"
        # Ensure LF line endings for Linux compatibility
        with open(path, 'w', encoding='utf-8', newline='\n') as f:
            f.write("\n".join(lines) + "\n")
        logger.info(f"Generated script: {path}")
