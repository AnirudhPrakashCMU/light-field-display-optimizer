# Light Field Display Optimizer

Enhanced PyTorch light field display optimizer with multi-ray sampling and spherical checkerboard scene support.

## Files

### Core Scripts
- **`spherical_checkerboard_raytracer.py`** - Spherical checkerboard ray tracer with realistic eye optics
- **`final_tests/optimizer.py`** - Complete multi-scene light field display optimizer
- **`final_tests/quick_checkerboard_test.py`** - Quick validation test (10 iterations)

### RunPod Version
- **`final_tests/RUNPOD_JUPYTER_COMPLETE.py`** - Enhanced optimizer for RunPod Jupyter with comprehensive debug outputs

## Features

### Optical Physics
- Multi-ray sub-aperture sampling for realistic depth-of-field
- Eye lens accommodation modeling
- Tunable focus lens simulation
- Circular microlens array processing
- MATLAB-compatible spherical coordinate mapping

### Scenes
- Basic geometric shapes
- Complex multi-object scenes
- Stick figure scenes
- Layered depth scenes
- Office environments
- Nature scenes
- **Spherical checkerboard** (MATLAB-compatible)

### Outputs
- Training progress GIFs
- Focal length sweep animations
- Eye movement parallax demonstrations
- Individual focal plane visualizations
- Comprehensive result archives

## Usage

### Quick Test (10 iterations, ~2 minutes)
```python
python quick_checkerboard_test.py
```

### Full Optimizer (7 scenes, ~1-2 hours)  
```python
python optimizer.py
```

### RunPod Jupyter (Enhanced with debug outputs)
1. Open RunPod Jupyter interface
2. Copy content from `RUNPOD_JUPYTER_COMPLETE.py`
3. Paste into notebook cell and run

## System Requirements
- NVIDIA GPU with CUDA support
- PyTorch with CUDA
- 10-80GB GPU memory (depending on configuration)
- Python packages: torch, matplotlib, pillow, opencv-python, tqdm, numpy

## Output Structure
```
results/
├── iteration_progress/     # Every iteration progress
├── focal_length_views/     # Individual focal plane views
├── gifs/                   # Training progress, focal sweep, eye movement
└── final_results/          # Complete results and archives
```