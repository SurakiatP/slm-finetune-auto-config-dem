#!/bin/bash
# setup_vast.sh - Automates the installation of dependencies for SLM Auto Config on Vast.ai

# --- Add Self-Logging ---
LOG_FILE="setup_vast.log"
exec > >(tee -i "$LOG_FILE") 2>&1
# ------------------------

echo "🛠 Starting Vast.ai Environment Setup..."

# 1. Update and install system dependencies
echo "📦 Updating system packages..."
sudo apt-get update
sudo apt-get install -y rsync git python3-pip htop

# 2. Install Python dependencies
echo "py 🐍 Installing Python packages..."
# We install the same specific versions as our local requirements.txt
pip3 install "oumi[gpu,tune]"
pip3 install pydantic==2.12.5 scikit-learn==1.6.1 PyYAML==6.0.2 "pandas>=2.3.0"
pip3 install fpdf2==2.8.2 matplotlib==3.10.0 seaborn==0.13.2 gradio

# 3. Create project structure on remote
mkdir -p ~/slm-auto-config/runs

# 4. Verify Oumi installation
echo "🔍 Verifying Oumi installation..."
if python3 -c "import oumi; print('Oumi is ready')" &> /dev/null
then
    echo "✅ Oumi is installed and importable."
else
    echo "❌ Oumi installation check failed."
fi

echo "--------------------------------------------------------"
echo "✅ Setup Complete! Environment is ready for training."
echo "You can now use 'sync_to_vast.sh' from your local machine."
echo "--------------------------------------------------------"
