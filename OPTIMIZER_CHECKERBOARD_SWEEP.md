# Checkerboard Density Sweep Optimizer

## Overview
Modified `standalone_optimizer.py` to run checkerboard density sweep optimization (26x26 to 62x62 squares).

## Key Changes

### 1. Ground Truth Configuration
- **1 ray per pixel** for ground truth generation (perfect pinhole camera)
- `samples_per_pixel_override = 1` (line 29)

### 2. Spherical Checkerboard
- Added `square_size` parameter to `SphericalCheckerboard` class
- Supports variable checkerboard densities (26x26, 28x28, ..., 60x60)

### 3. Removed Scenes
- Deleted all other scenes (basic, complex, stick_figure, layered, office, nature, textured_basic)
- Only `create_spherical_checkerboard(square_size)` remains

### 4. New Main Function
- `run_checkerboard_density_sweep()` - Sweeps through checkerboard densities
- Runs 50 iterations of optimization per checkerboard
- Creates GIF of optimized eye views across all densities
- Frame repetition (10x) for slower playback

## Optimization Verification ✅

**This is 100% REAL OPTIMIZATION** - Not mocked or faked:

```python
# Line 904: Real optimizer
optimizer = optim.AdamW(display_system.parameters(), lr=0.02)

# Line 912: Ground truth via ray tracing
target_image = render_eye_view_target(eye_position, eye_focal_length, scene_objects, resolution)

# Line 926-928: Forward pass through optical system
simulated_image = render_eye_view_through_display(
    eye_position, eye_focal_length, display_system, resolution
)

# Line 931: MSE loss
loss = torch.mean((simulated_image - target_image) ** 2)

# Line 932: BACKPROPAGATION through ray tracing
loss.backward()

# Line 934: GRADIENT DESCENT update
optimizer.step()
```

### Gradient Flow
1. Ray tracing through optical system (`render_eye_view_through_display`)
2. MSE loss computation
3. Backprop through ALL ray tracing operations
4. Gradients computed for ALL 8 display images
5. AdamW updates display pixel values
6. Display images clamped to [0, 1]

## Outputs

### Per Checkerboard (e.g., 26x26, 28x28, ..., 60x60)
- `progress_all_frames.gif` - Optimization progression (50 frames)
- `what_displays_show.png` - All 8 optimized display images
- `what_eye_sees.png` - Eye views through each display
- `focal_sweep_through_display.gif` - Focal length sweep
- `eye_movement_through_display.gif` - Eye movement sweep
- `REAL_scene_focal_sweep.gif` - Ground truth focal sweep
- `REAL_scene_eye_movement.gif` - Ground truth eye movement

### Main Output
- `optimized_eye_view_sweep.gif` - Eye view at nominal position (x=2mm, f=30mm) across all checkerboard densities

## Parameters
- **Iterations**: 50 per checkerboard
- **Resolution**: 128x128 for rendering
- **Display resolution**: 512x512 (8 planes)
- **Checkerboard range**: 26x26 to 60x60 (increment by 2)
- **Ground truth**: 1 ray per pixel
- **Learning rate**: 0.02 (AdamW)
- **Nominal viewpoint**: x=2mm, f=30mm

## Comparison with Competitor & Ground Truth

| System | Method | Rays/Pixel |
|--------|--------|-----------|
| **Ground Truth** | Direct eye→scene ray tracing | 1 (perfect) |
| **Competitor** | Inverse rendering (MLA+display) | N/A |
| **Optimizer** | Gradient descent optimization | 1 (target), 4 (simulated) |

## File Location
`/Users/anirudhprakash/Desktop/Projects/Light Field Display/final_code/optimizer/standalone_optimizer.py`

## Run Command
```bash
python standalone_optimizer.py
```

## Expected Runtime
~1-2 hours for full sweep (18 checkerboards × 50 iterations)
