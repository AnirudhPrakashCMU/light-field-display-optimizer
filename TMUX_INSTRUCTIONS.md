# RunPod Instructions - Light Field Display Systems

## Run All Three Systems in One Command

This script runs all three light field display systems **sequentially** in a single tmux session:
1. **Ground Truth** - Direct ray tracing (~5-10 min)
2. **Competitor** - Inverse rendering (~10-20 min)
3. **Optimizer** - Gradient-based optimization (~4-6 hours)

Total runtime: **~5-7 hours** for complete comparison.

---

## Quick Start

### Step 1: Create RunPod Instance
- Go to RunPod
- Create a pod with **CUDA 11.8+** and **PyTorch** template
- Use **A100** or **A6000** GPU for best performance

### Step 2: Open Web Terminal
- Click **Connect → Web Terminal**

### Step 3: Start tmux Session
```bash
tmux new -s lightfield
```

### Step 4: Run Everything
```bash
cd /workspace
git clone https://github.com/AnirudhPrakashCMU/light-field-display-optimizer.git
cd light-field-display-optimizer
chmod +x run_all_systems.sh
./run_all_systems.sh
```

### Step 5: Detach from tmux (Safe to Close Browser)
Press: **Ctrl + B** then **D**

✅ **You can now close your browser tab safely!**

---

## What Happens

The script will:
1. ✅ Install all dependencies (PyTorch, matplotlib, etc.)
2. 🌍 Run ground truth raytracer (8 rays/pixel, 512×512)
3. 🏁 Run inverse renderer competitor (8 rays/pixel, 1024×1024 displays)
4. 🎯 Run gradient-based optimizer (10 focal planes, 100 iterations/checkerboard)
5. 📦 Create combined ZIP archive
6. ☁️ Upload everything to catbox.moe
7. 🎉 Display permanent download URL

---

## Check Progress Later

### Reattach to tmux:
```bash
tmux attach -t lightfield
```

### Check progress without reattaching:
```bash
# See what's currently running
ps aux | grep python

# Check latest logs
tail -f /workspace/all_results_*/ground_truth/ground_truth_log.txt
tail -f /workspace/all_results_*/competitor/competitor_log.txt
tail -f /workspace/all_results_*/optimizer/optimizer_log.txt

# Get download URL
cat /workspace/all_results_*/upload_info.json
```

---

## Output Structure

```
/workspace/all_results_YYYYMMDD_HHMMSS/
├── ground_truth/
│   ├── ground_truth_log.txt
│   ├── comparatives/
│   │   └── ground_truth_density_sweep.gif
│   └── results/
├── competitor/
│   ├── competitor_log.txt
│   ├── comparatives/
│   │   └── competitor_density_sweep.gif
│   └── outputs_ft_python/
│       └── debugging_outputs/
├── optimizer/
│   ├── optimizer_log.txt
│   ├── comparatives/
│   └── results/
├── all_systems_results.zip
└── upload_info.json (contains catbox URL)
```

---

## System Specifications

All three systems use **matched optical parameters** for fair comparison:

| Parameter | Value |
|-----------|-------|
| MLA distance | 80mm |
| MLA focal length | 1.0mm |
| MLA pitch | 0.4mm |
| Display distance | 82mm |
| Tunable lens distance | 50mm |
| Rendering resolution | 512×512 |
| Display resolution | 1024×1024 |
| Rays per pixel | 8 |
| Focal planes (optimizer) | 10 (linear in 1/f) |
| Checkerboard range | 26×26 to 60×60 |

---

## Expected Runtimes (A100 GPU)

- **Ground Truth**: 5-10 minutes
- **Competitor**: 10-20 minutes
- **Optimizer**: 4-6 hours (18 checkerboards × 100 iterations)
- **Total**: ~5-7 hours

---

## Troubleshooting

### If script fails:
1. Check the log files in the output directory
2. Verify GPU is available: `nvidia-smi`
3. Check Python version: `python3 --version` (should be 3.8+)
4. Re-run the script - it will create a new timestamped directory

### Kill stuck session:
```bash
tmux kill-session -t lightfield
```

### Restart from scratch:
```bash
# Kill any running Python processes
pkill -9 python

# Start fresh
tmux new -s lightfield
cd /workspace/light-field-display-optimizer
git pull
./run_all_systems.sh
```

---

## What Each System Does

### 🌍 Ground Truth
- Pure ray tracing from eye to physical spherical checkerboard
- No display, no MLA - just the physical scene
- 8 rays per pixel for smooth rendering
- Output: `ground_truth_density_sweep.gif`

### 🏁 Competitor (Inverse Renderer)
- **Display generation**: Inverse ray tracing (Display → MLA → Scene)
- **Viewing**: Forward ray tracing (Eye → Tunable Lens → MLA → Display)
- 8 rays/pixel with stratified jittering
- Output: `competitor_density_sweep.gif` + debug outputs

### 🎯 Optimizer
- Gradient-based optimization with PyTorch
- Real backpropagation through differentiable ray tracing
- 10 focal planes (linear in 1/f from 10mm to 100mm)
- 100 iterations per checkerboard density
- Output: Optimized display patterns + eye view GIFs

---

## After Completion

You'll get a **permanent catbox.moe URL** containing:
- All three system outputs
- Density sweep GIFs showing checkerboard resolution limits
- Complete logs for each system
- Debug outputs and intermediate results

The URL will be displayed at the end and saved in `upload_info.json`.

---

## Notes

- The script installs dependencies automatically
- All systems run on the same GPU sequentially
- Results are uploaded automatically to catbox.moe
- You can safely close your browser after detaching from tmux
- The tmux session keeps running even if you disconnect
