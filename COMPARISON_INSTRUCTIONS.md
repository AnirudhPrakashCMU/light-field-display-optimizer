# Running Competitor vs Optimizer Comparison

## Identical Configuration

Both systems now use **exactly the same configuration**:

| Parameter | Value |
|-----------|-------|
| Sphere center | 80mm from eye |
| Sphere radius | 10mm |
| Rays per pixel | 8 (stratified jittering) |
| Rendering resolution | 512×512 |
| Display resolution | 1024×1024 |
| Eye position | x=0mm, y=0mm, z=0mm |
| Eye focal length | 30mm |
| Checkerboard range | 25, 30, 35, 40, 45, 50, 55, 60 |

**Key Difference**:
- **Competitor**: Inverse ray tracing (sphere → MLA → display)
- **Optimizer**: Gradient-based optimization with forward rendering (display → MLA → eye)

---

## Local Run (macOS/Linux)

```bash
cd /Users/anirudhprakash/Desktop/Projects/Light\ Field\ Display/final_code/optimizer

# Run both systems sequentially
./run_competitor_and_optimizer.sh
```

**Expected Runtime**:
- Competitor: ~10-15 minutes
- Optimizer: ~2 hours (8 checkerboards × 50 iterations)
- **Total**: ~2-2.5 hours

---

## TMUX Instructions (for long-running sessions)

### Step 1: Start tmux session
```bash
tmux new -s comparison
```

### Step 2: Run the script
```bash
cd /Users/anirudhprakash/Desktop/Projects/Light\ Field\ Display/final_code/optimizer
./run_competitor_and_optimizer.sh
```

### Step 3: Detach from tmux (safe to close terminal)
Press: **Ctrl + B** then **D**

### Step 4: Reattach later to check progress
```bash
tmux attach -t comparison
```

### Step 5: Check progress without attaching
```bash
# Check latest results directory
ls -lt comparison_results_* | head -1

# Monitor logs in real-time
tail -f comparison_results_*/competitor_log.txt
tail -f comparison_results_*/optimizer_log.txt
```

---

## RunPod Cloud Execution

For faster execution on A100 GPU, use the existing unified script:

```bash
tmux new -s lightfield
cd /workspace
git clone https://github.com/AnirudhPrakashCMU/light-field-display-optimizer.git
cd light-field-display-optimizer
chmod +x run_all_systems.sh
./run_all_systems.sh
# Press Ctrl+B then D to detach
```

This will run:
1. Ground truth (~5-10 min)
2. Competitor (~10-20 min)
3. Optimizer (~2 hours)

All results uploaded to catbox.moe automatically.

---

## Outputs

After completion, you'll find in `comparison_results_TIMESTAMP/`:

```
comparison_results_YYYYMMDD_HHMMSS/
├── competitor_density_sweep.gif          # Competitor eye view GIF
├── competitor_outputs/                   # Competitor debug outputs
│   ├── checkerboard_*_display.png       # Display patterns
│   ├── checkerboard_*_views_*.png       # Eye views
│   └── checkerboard_*_*.gif             # Animations
├── optimizer_results/                    # Optimizer outputs
│   └── scenes/                          # Per-checkerboard results
├── competitor_log.txt                    # Competitor full log
└── optimizer_log.txt                     # Optimizer full log
```

---

## Verification Checklist

✅ **Before running, verify both systems match:**

**Competitor** (`inverse_renderer_competitor.py`):
```python
# Line 43-46: MLA at 80mm
self.z0 = 80.0
self.disp_z0 = 82.0

# Line 80-81: Sphere at 80mm, radius 10mm
self.center = torch.tensor([0.0, 0.0, mla.z0], device=device)
self.radius = mla.width / 2.0  # = 10mm

# Line 369: 8 rays per pixel
def render_camera_view(..., samples_per_pixel=8)

# Line 617: Eye position x=0mm, f=30mm
nominal_view = render_camera_view(mla, [0, 0, 0], cam_res=512,
                                  tunable_focal_length=30.0)
```

**Optimizer** (`standalone_optimizer.py`):
```python
# Line 29: 8 rays per pixel
samples_per_pixel_override = 8

# Line 473-474: Sphere at 80mm, radius 10mm
center=torch.tensor([0.0, 0.0, 80.0], device=device)
radius=10.0

# Line 1199-1200: Eye position x=0mm, f=30mm
eye_position = torch.tensor([0.0, 0.0, 0.0], device=device)
eye_focal_length = 30.0

# Line 1272: Rendering resolution 512x512
resolution = 512

# Line 1271: 50 iterations per checkerboard
iterations = 50
```

---

## Quick Test (Single Checkerboard)

To test with just one checkerboard (faster):

1. Edit `inverse_renderer_competitor.py` line 635:
   ```python
   for num_squares in [40]:  # Single test
   ```

2. Edit `standalone_optimizer.py` line 1288:
   ```python
   for num_squares in [40]:  # Single test
   ```

3. Run: `./run_competitor_and_optimizer.sh`

This will complete in ~20 minutes instead of 2+ hours.
