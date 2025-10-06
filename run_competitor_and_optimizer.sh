#!/bin/bash

# Run Competitor and Optimizer with IDENTICAL configurations
# Both use: sphere at 80mm with 10mm radius, 8 rays/pixel, 512x512 rendering

echo "🔬 RUNNING COMPETITOR AND OPTIMIZER WITH IDENTICAL CONFIGURATIONS"
echo "=================================================================="
echo ""
echo "Shared Configuration:"
echo "  • Sphere: center at 80mm, radius 10mm"
echo "  • Rays per pixel: 8 (with stratified jittering)"
echo "  • Rendering resolution: 512x512"
echo "  • Display resolution: 1024x1024"
echo "  • Eye position: x=0mm, focal length f=30mm"
echo "  • Checkerboard sweep: 25, 30, 35, 40, 45, 50, 55, 60"
echo ""
echo "Different rendering approaches:"
echo "  • Competitor: Inverse ray tracing (sphere → MLA → display)"
echo "  • Optimizer: Gradient-based optimization with forward rendering"
echo ""

# Create timestamped results directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="comparison_results_${TIMESTAMP}"
mkdir -p ${RESULTS_DIR}

echo "📁 Results will be saved to: ${RESULTS_DIR}"
echo ""

# ============================================
# COMPETITOR
# ============================================
echo "════════════════════════════════════════"
echo "🏁 RUNNING COMPETITOR (Inverse Renderer)"
echo "════════════════════════════════════════"
echo ""

python3 inverse_renderer_competitor.py > ${RESULTS_DIR}/competitor_log.txt 2>&1

echo "✅ Competitor complete"
echo ""

# Copy competitor results
cp comparatives/competitor_density_sweep.gif ${RESULTS_DIR}/
cp -r outputs_ft_python/debugging_outputs ${RESULTS_DIR}/competitor_outputs

# ============================================
# OPTIMIZER
# ============================================
echo "════════════════════════════════════════"
echo "🎯 RUNNING OPTIMIZER (Gradient-based)"
echo "════════════════════════════════════════"
echo "⏰ Started at: $(date)"
echo "Expected runtime: ~2 hours (8 checkerboards × 50 iterations)"
echo ""

python3 standalone_optimizer.py > ${RESULTS_DIR}/optimizer_log.txt 2>&1

echo "✅ Optimizer complete at: $(date)"
echo ""

# Copy optimizer results
cp -r results ${RESULTS_DIR}/optimizer_results 2>/dev/null || true
cp -r comparatives ${RESULTS_DIR}/optimizer_comparatives 2>/dev/null || true

# ============================================
# SUMMARY
# ============================================
echo "════════════════════════════════════════"
echo "✅ BOTH SYSTEMS COMPLETE!"
echo "════════════════════════════════════════"
echo ""
echo "📁 Results in: ${RESULTS_DIR}/"
echo ""
echo "Outputs:"
echo "  • competitor_density_sweep.gif - Competitor's eye view sweep"
echo "  • competitor_outputs/ - Competitor debug outputs"
echo "  • optimizer_results/ - Optimizer outputs"
echo "  • competitor_log.txt - Competitor full log"
echo "  • optimizer_log.txt - Optimizer full log"
echo ""
echo "Both systems used IDENTICAL configurations:"
echo "  ✓ Sphere at 80mm, radius 10mm"
echo "  ✓ 8 rays per pixel"
echo "  ✓ 512x512 rendering, 1024x1024 displays"
echo "  ✓ Same viewing parameters (x=0mm, f=30mm)"
echo ""
