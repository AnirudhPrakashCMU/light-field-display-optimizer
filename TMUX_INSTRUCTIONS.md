# RunPod Web Terminal Instructions - Light Field Display Systems

## Run All Three Systems in tmux (Safe to Close Browser)

This repository contains three light field display systems:
1. **Ground Truth** - Direct ray tracing (eye → physical scene)
2. **Competitor** - Inverse rendering with MLA + tunable lens
3. **Optimizer** - Gradient-based optimization to learn display patterns

All systems output to catbox.moe with permanent download URLs.

---

## System 1: Ground Truth Raytracer

### Step 1: Open RunPod Web Terminal
- Go to your RunPod pod dashboard
- Click **Connect → Web Terminal**

### Step 2: Start tmux Session
```bash
tmux new -s ground_truth
```

### Step 3: Run Ground Truth System
```bash
cd /workspace
git clone https://github.com/AnirudhPrakashCMU/light-field-display-optimizer.git
cd light-field-display-optimizer
chmod +x run_ground_truth.sh
./run_ground_truth.sh
```

### Step 4: Detach from tmux (Safe to Close Tab)
Press: **Ctrl + B** then **D**

✅ **You can now close your browser tab safely**

### Expected Output:
- **Resolution**: 512x512
- **Rays per pixel**: 8
- **Checkerboard sweep**: 26x26 to 60x60
- **Runtime**: ~5-10 minutes
- **Output**: ZIP with all results uploaded to catbox.moe

---

## System 2: Inverse Renderer Competitor

### Step 1: Start tmux Session
```bash
tmux new -s competitor
```

### Step 2: Run Competitor System
```bash
cd /workspace
git clone https://github.com/AnirudhPrakashCMU/light-field-display-optimizer.git
cd light-field-display-optimizer
chmod +x run_competitor.sh
./run_competitor.sh
```

### Step 3: Detach from tmux
Press: **Ctrl + B** then **D**

### Expected Output:
- **Display resolution**: 1024x1024
- **Rendering resolution**: 512x512
- **Rays per pixel**: 8 (with stratified jittering)
- **MLA config**: 80mm distance, 0.4mm pitch
- **Checkerboard sweep**: 26x26 to 60x60
- **Runtime**: ~10-20 minutes
- **Output**: ZIP with all results uploaded to catbox.moe

---

## System 3: Light Field Optimizer

### Step 1: Start tmux Session
```bash
tmux new -s optimizer
```

### Step 2: Run Optimizer System
```bash
cd /workspace
git clone https://github.com/AnirudhPrakashCMU/light-field-display-optimizer.git
cd light-field-display-optimizer
chmod +x run_optimizer.sh
./run_optimizer.sh
```

### Step 3: Detach from tmux
Press: **Ctrl + B** then **D**

### Expected Output:
- **Display resolution**: 1024x1024
- **Rendering resolution**: 512x512
- **Rays per pixel**: 8
- **Focal planes**: 10 (linear in 1/f from 10mm to 100mm)
- **Checkerboard sweep**: 26x26 to 60x60
- **Iterations**: 100 per checkerboard
- **Runtime**: ~4-6 hours (A100 optimized)
- **Output**: ZIP with all results uploaded to catbox.moe

---

## Check Progress Later

### Reattach to tmux session:
```bash
# Ground truth
tmux attach -t ground_truth

# Competitor
tmux attach -t competitor

# Optimizer
tmux attach -t optimizer
```

### Or check logs without reattaching:
```bash
# Ground truth
tail -f /workspace/ground_truth_results_*/ground_truth_log.txt

# Competitor
tail -f /workspace/competitor_results_*/competitor_log.txt

# Optimizer
tail -f /workspace/optimizer_results_*/optimizer_log.txt
```

### Get download URLs:
```bash
# Ground truth
cat /workspace/ground_truth_results_*/upload_info.json

# Competitor
cat /workspace/competitor_results_*/upload_info.json

# Optimizer
cat /workspace/optimizer_results_*/upload_info.json
```

---

## Optical Configuration (Consistent Across All Systems)

All three systems use matched optical parameters for fair comparison:

| Parameter | Value |
|-----------|-------|
| MLA distance | 80mm |
| MLA focal length | 1.0mm |
| MLA pitch | 0.4mm |
| Display distance | 82mm |
| Tunable lens distance | 50mm |
| Rendering resolution | 512x512 |
| Display resolution (optimizer/competitor) | 1024x1024 |
| Rays per pixel | 8 |

---

## What Each System Does

### Ground Truth
- Pure ray tracing from eye to physical spherical checkerboard
- No display, no MLA - just the scene
- Single ray per pixel originally, now 8 rays for consistency
- Generates: `ground_truth_density_sweep.gif`

### Competitor
- Inverse ray tracing: Display → MLA → Scene
- Forward ray tracing: Eye → Tunable Lens → MLA → Display
- 8 rays per pixel with stratified jittering
- Generates: `competitor_density_sweep.gif` + debug outputs

### Optimizer
- Gradient-based optimization with PyTorch
- Real backpropagation through ray tracing
- 10 focal planes (linear in 1/f)
- 100 iterations per checkerboard
- Generates: Optimized display patterns + eye view GIFs

---

## Safety Features

- **Timestamped output directories**: Never overwrites previous runs
- **Complete logging**: All output saved to log files
- **JSON metadata**: Download URLs saved locally
- **tmux protection**: Survives browser disconnection
- **Automatic upload**: All results uploaded to catbox.moe
- **ZIP archives**: All outputs packaged for easy download

---

## Troubleshooting

### Check if process is still running:
```bash
ps aux | grep python
```

### Kill a stuck session:
```bash
tmux kill-session -t ground_truth
tmux kill-session -t competitor
tmux kill-session -t optimizer
```

### Check GPU usage:
```bash
nvidia-smi
```

### Re-run a system:
Just run the corresponding script again - it will create a new timestamped directory.
