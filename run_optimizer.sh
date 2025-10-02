#!/bin/bash

# Light Field Optimizer System Runner with Catbox Upload
# Run this in tmux to safely detach and close browser tabs

echo "🎯 LIGHT FIELD OPTIMIZER - TMUX RUNNER"
echo "Safe to close browser tab after starting"

# Create timestamped output directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="/workspace/optimizer_results_${TIMESTAMP}"
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

# Run optimizer
echo "🚀 Starting light field optimizer..."
echo "⏰ Started at: $(date)"

python3 standalone_optimizer.py > ${OUTPUT_DIR}/optimizer_log.txt 2>&1

# Copy results to output directory
echo "📦 Copying results..."
cp -r comparatives ${OUTPUT_DIR}/ 2>/dev/null || true
cp -r results ${OUTPUT_DIR}/ 2>/dev/null || true

# Create ZIP archive
echo "🗜️ Creating ZIP archive..."
cd ${OUTPUT_DIR}
zip -r optimizer_results.zip . -q

# Upload to catbox.moe
echo "☁️ Uploading ZIP to catbox.moe..."
CATBOX_URL=$(curl -F "reqtype=fileupload" -F "fileToUpload=@optimizer_results.zip" https://catbox.moe/user/api.php)

# Save results
echo "{" > upload_info.json
echo "  \"zip_url\": \"${CATBOX_URL}\"," >> upload_info.json
echo "  \"timestamp\": \"${TIMESTAMP}\"," >> upload_info.json
echo "  \"system\": \"optimizer\"" >> upload_info.json
echo "}" >> upload_info.json

echo ""
echo "✅ Optimizer complete!"
echo "📁 Results saved to: ${OUTPUT_DIR}"
echo "🌐 ZIP download URL: ${CATBOX_URL}"
echo "📋 Check optimizer_log.txt for full output"
echo ""
echo "Copy this URL:"
echo "${CATBOX_URL}"
