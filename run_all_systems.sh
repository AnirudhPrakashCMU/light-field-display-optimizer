#!/bin/bash

# Light Field Display - Run All Three Systems Sequentially
# Safe to run in tmux and detach

echo "🚀 LIGHT FIELD DISPLAY - ALL SYSTEMS RUNNER"
echo "=============================================="
echo ""
echo "This will run sequentially:"
echo "  1. Ground Truth Raytracer (~5-10 min)"
echo "  2. Inverse Renderer Competitor (~10-20 min)"
echo "  3. Light Field Optimizer (~4-6 hours)"
echo ""
echo "Safe to close browser tab after starting in tmux"
echo ""

# Create timestamped output directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_OUTPUT_DIR="/workspace/all_results_${TIMESTAMP}"
mkdir -p ${MAIN_OUTPUT_DIR}

echo "📁 All results will be saved to: ${MAIN_OUTPUT_DIR}"
echo ""

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
    echo ""
fi

cd light-field-display-optimizer

# Pull latest changes
echo "🔄 Pulling latest code..."
git pull
echo ""

# Install dependencies ONCE at the beginning
echo "📦 Installing dependencies..."
pip3 install --upgrade pip
pip3 install torch torchvision torchaudio
pip3 install matplotlib pillow requests numpy
echo ""
echo "✅ Dependencies installed"
echo ""

# ============================================
# SYSTEM 1: Ground Truth
# ============================================
echo "════════════════════════════════════════"
echo "🌍 SYSTEM 1: GROUND TRUTH RAYTRACER"
echo "════════════════════════════════════════"
echo "⏰ Started at: $(date)"
echo ""

GT_OUTPUT_DIR="${MAIN_OUTPUT_DIR}/ground_truth"
mkdir -p ${GT_OUTPUT_DIR}

python3 spherical_checkerboard_raytracer.py > ${GT_OUTPUT_DIR}/ground_truth_log.txt 2>&1

# Copy results
echo "📦 Copying ground truth results..."
cp -r comparatives ${GT_OUTPUT_DIR}/ 2>/dev/null || true
cp -r results ${GT_OUTPUT_DIR}/ 2>/dev/null || true

echo "✅ Ground truth complete at: $(date)"
echo ""

# ============================================
# SYSTEM 2: Competitor
# ============================================
echo "════════════════════════════════════════"
echo "🏁 SYSTEM 2: INVERSE RENDERER COMPETITOR"
echo "════════════════════════════════════════"
echo "⏰ Started at: $(date)"
echo ""

COMP_OUTPUT_DIR="${MAIN_OUTPUT_DIR}/competitor"
mkdir -p ${COMP_OUTPUT_DIR}

python3 inverse_renderer_competitor.py > ${COMP_OUTPUT_DIR}/competitor_log.txt 2>&1

# Copy results
echo "📦 Copying competitor results..."
cp -r comparatives ${COMP_OUTPUT_DIR}/ 2>/dev/null || true
cp -r outputs_ft_python ${COMP_OUTPUT_DIR}/ 2>/dev/null || true

echo "✅ Competitor complete at: $(date)"
echo ""

# ============================================
# SYSTEM 3: Optimizer
# ============================================
echo "════════════════════════════════════════"
echo "🎯 SYSTEM 3: LIGHT FIELD OPTIMIZER"
echo "════════════════════════════════════════"
echo "⏰ Started at: $(date)"
echo ""

OPT_OUTPUT_DIR="${MAIN_OUTPUT_DIR}/optimizer"
mkdir -p ${OPT_OUTPUT_DIR}

python3 standalone_optimizer.py > ${OPT_OUTPUT_DIR}/optimizer_log.txt 2>&1

# Copy results
echo "📦 Copying optimizer results..."
cp -r comparatives ${OPT_OUTPUT_DIR}/ 2>/dev/null || true
cp -r results ${OPT_OUTPUT_DIR}/ 2>/dev/null || true

echo "✅ Optimizer complete at: $(date)"
echo ""

# ============================================
# Create combined ZIP and upload
# ============================================
echo "════════════════════════════════════════"
echo "📦 CREATING COMBINED ZIP ARCHIVE"
echo "════════════════════════════════════════"

cd ${MAIN_OUTPUT_DIR}
zip -r all_systems_results.zip . -q

echo "✅ ZIP created: all_systems_results.zip"
echo ""

# Upload to catbox.moe
echo "☁️  Uploading to catbox.moe..."
CATBOX_URL=$(curl -F "reqtype=fileupload" -F "fileToUpload=@all_systems_results.zip" https://catbox.moe/user/api.php)

# Save results
cat > ${MAIN_OUTPUT_DIR}/upload_info.json <<EOF
{
  "zip_url": "${CATBOX_URL}",
  "timestamp": "${TIMESTAMP}",
  "systems": ["ground_truth", "competitor", "optimizer"]
}
EOF

# ============================================
# Final Summary
# ============================================
echo ""
echo "════════════════════════════════════════"
echo "✅ ALL SYSTEMS COMPLETE!"
echo "════════════════════════════════════════"
echo ""
echo "📁 Local results: ${MAIN_OUTPUT_DIR}"
echo "🌐 Download URL: ${CATBOX_URL}"
echo ""
echo "📂 Contents:"
echo "  • ground_truth/    - Ground truth raytracer results"
echo "  • competitor/      - Inverse renderer competitor results"
echo "  • optimizer/       - Light field optimizer results"
echo "  • upload_info.json - Download URL metadata"
echo ""
echo "⏰ Finished at: $(date)"
echo ""
echo "════════════════════════════════════════"
echo "COPY THIS DOWNLOAD URL:"
echo "${CATBOX_URL}"
echo "════════════════════════════════════════"
