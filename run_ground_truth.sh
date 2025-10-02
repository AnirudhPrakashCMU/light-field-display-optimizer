#!/bin/bash

# Ground Truth System Runner with Catbox Upload
# Run this in tmux to safely detach and close browser tabs

echo "🌍 GROUND TRUTH RAYTRACER - TMUX RUNNER"
echo "Safe to close browser tab after starting"

# Create timestamped output directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="/workspace/ground_truth_results_${TIMESTAMP}"
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

# Run ground truth raytracer
echo "🚀 Starting ground truth raytracer..."
echo "⏰ Started at: $(date)"

python3 spherical_checkerboard_raytracer.py > ${OUTPUT_DIR}/ground_truth_log.txt 2>&1

# Copy results to output directory
echo "📦 Copying results..."
cp -r comparatives ${OUTPUT_DIR}/
cp -r results ${OUTPUT_DIR}/

# Create ZIP archive
echo "🗜️ Creating ZIP archive..."
cd ${OUTPUT_DIR}
zip -r ground_truth_results.zip . -q

# Upload to catbox.moe
echo "☁️ Uploading ZIP to catbox.moe..."
CATBOX_URL=$(curl -F "reqtype=fileupload" -F "fileToUpload=@ground_truth_results.zip" https://catbox.moe/user/api.php)

# Save results
echo "{" > upload_info.json
echo "  \"zip_url\": \"${CATBOX_URL}\"," >> upload_info.json
echo "  \"timestamp\": \"${TIMESTAMP}\"," >> upload_info.json
echo "  \"system\": \"ground_truth\"" >> upload_info.json
echo "}" >> upload_info.json

echo ""
echo "✅ Ground truth complete!"
echo "📁 Results saved to: ${OUTPUT_DIR}"
echo "🌐 ZIP download URL: ${CATBOX_URL}"
echo "📋 Check ground_truth_log.txt for full output"
echo ""
echo "Copy this URL:"
echo "${CATBOX_URL}"
