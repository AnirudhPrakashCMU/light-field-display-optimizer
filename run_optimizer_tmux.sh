#!/bin/bash

# RunPod Web Terminal Light Field Optimizer Runner
# Run this in tmux to safely detach and close browser tabs

echo "🚀 LIGHT FIELD OPTIMIZER - TMUX RUNNER"
echo "Safe to close browser tab after starting"

# Create timestamped output directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="/workspace/optimization_results_${TIMESTAMP}"
mkdir -p ${OUTPUT_DIR}

echo "📁 Results will be saved to: ${OUTPUT_DIR}"

# Set up environment
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

# Change to workspace
cd /workspace

# Clone repository if not exists
if [ ! -d "light-field-display-optimizer" ]; then
    echo "📥 Cloning repository..."
    git clone https://github.com/AnirudhPrakashCMU/light-field-display-optimizer.git
fi

cd light-field-display-optimizer

# Pull latest changes
echo "🔄 Pulling latest code..."
git pull

# Install dependencies
echo "📦 Installing dependencies..."
pip install torch torchvision matplotlib pillow requests numpy

# Run the optimization
echo "🚀 Starting light field optimization..."
echo "⏰ Started at: $(date)"

# Run the standalone optimizer
python3 standalone_optimizer.py > ${OUTPUT_DIR}/optimization_log.txt 2>&1

echo "✅ Optimization complete!"
echo "📁 Results saved to: ${OUTPUT_DIR}"
echo "📋 Check optimization_log.txt for full output"
echo "📥 Check optimization_results.json for download URLs"