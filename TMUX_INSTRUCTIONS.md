# RunPod Web Terminal Instructions - Light Field Optimizer

## Run the Complete Light Field Optimizer in tmux (Safe to Close Browser)

### Step 1: Open RunPod Web Terminal
- Go to your RunPod pod dashboard
- Click **Connect → Web Terminal**

### Step 2: Start tmux Session
```bash
tmux new -s lightfield
```

### Step 3: Run the Optimizer
```bash
cd /workspace
git clone https://github.com/AnirudhPrakashCMU/light-field-display-optimizer.git
cd light-field-display-optimizer
chmod +x run_optimizer_tmux.sh
./run_optimizer_tmux.sh
```

### Step 4: Detach from tmux (Safe to Close Tab)
Press: **Ctrl + B** then **D**

✅ **You can now close your browser tab safely**

### Step 5: Check Progress Later
When you come back:
```bash
tmux attach -t lightfield
```

Or just check the logs:
```bash
tail -f /workspace/optimization_results_*/optimization_log.txt
```

### Step 6: Get Results
Results will be in:
- `/workspace/optimization_results_[timestamp]/optimization_results.json` - All download URLs
- `/workspace/optimization_results_[timestamp]/optimization_log.txt` - Complete logs

### Alternative: Check Without Reattaching
```bash
# See if optimization is still running
ps aux | grep python

# Check latest results
ls -la /workspace/optimization_results_*/

# View latest log
tail -n 50 /workspace/optimization_results_*/optimization_log.txt
```

## What the Optimization Does
- **ALL 7 scenes optimized**: basic, complex, stick_figure, layered, office, nature, spherical_checkerboard
- **25 iterations per scene**: Every iteration tracked
- **5 outputs per scene**: Progress GIF, display images, eye views, focal sweep, eye movement
- **35 total download URLs**: All uploaded to catbox.moe
- **ACTUAL ray tracing**: Complete optical system simulation

## Expected Runtime
- **~20-30 minutes** for all 7 scenes
- **Progress updates** every 10 iterations
- **All results** uploaded with permanent URLs

## Safety Features
- **Timestamped output directory**: Never overwrites previous runs
- **Complete logging**: All output saved to log file
- **JSON results**: All download URLs saved locally
- **tmux protection**: Survives browser disconnection