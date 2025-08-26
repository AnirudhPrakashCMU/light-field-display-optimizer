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

# Run with all outputs saved
python3 -c "
import sys
sys.path.append('/workspace/light-field-display-optimizer')

# Import and run the handler directly
exec(open('rp_handler.py').read().replace('runpod.serverless.start', '# runpod.serverless.start'))

# Create job input
job_input = {
    'input': {
        'iterations': 25,
        'resolution': 128
    }
}

print('🎯 Running complete optimization...')
result = handler(job_input)

print('\\n' + '='*80)
print('📋 OPTIMIZATION COMPLETE!')
print('='*80)

if result.get('status') == 'success':
    print(f'✅ SUCCESS: {result.get(\"message\", \"\")}')
    print(f'📊 Scenes completed: {len(result.get(\"scenes_completed\", []))}')
    print(f'📥 Total download URLs: {len(result.get(\"all_download_urls\", {}))}')
    
    print(f'\\n📥 ALL DOWNLOAD URLS:')
    for name, url in result.get('all_download_urls', {}).items():
        print(f'   {name}: {url}')
    
    # Save results to file
    import json
    with open('${OUTPUT_DIR}/optimization_results.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f'\\n📋 Results saved to: ${OUTPUT_DIR}/optimization_results.json')
else:
    print(f'❌ FAILED: {result.get(\"message\", \"\")}')
    if 'traceback' in result:
        print(f'Error details: {result[\"traceback\"]}')

print(f'\\n⏰ Completed at: $(date)')
" > ${OUTPUT_DIR}/optimization_log.txt 2>&1

echo "✅ Optimization complete!"
echo "📁 Results saved to: ${OUTPUT_DIR}"
echo "📋 Check optimization_log.txt for full output"
echo "📥 Check optimization_results.json for download URLs"