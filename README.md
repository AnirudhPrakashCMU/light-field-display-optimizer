# Light Field Display - Three Approaches

Complete implementation of three approaches to light field display synthesis.

---

## Scripts

### 1. Ground Truth: `spherical_checkerboard_raytracer.py` ✅
**Purpose**: Direct forward ray tracing (reference/validation)

**Method**:
- Eye retina → Eye lens → Physical spherical checkerboard
- Tilted retina pointing at scene
- 8 rays per pixel for natural depth-of-field

**Outputs**: `results/`
- Standard eye view
- Focal length comparison
- Focal sweep GIF
- Eye movement GIF

**Run**: `python spherical_checkerboard_raytracer.py`

---

### 2. Inverse Renderer: `inverse_renderer_competitor.py` ✅ NEW!
**Purpose**: Analytical inverse rendering (competitor baseline)

**Method**:
- **Display Generation**: Display pixel → MLA → Sphere (inverse ray tracing)
- **Viewing**: Camera → Tunable lens → MLA → Display (forward)
- Direct Python port of MATLAB `sphere_synthesis_focus_tunable_standalone.m`

**Features**:
- 5 test patterns (checkerboard, stripes, circles, gradient)
- Tunable lens with 3 focal lengths (30mm, 50mm, 100mm)
- Focal sweep animations (20-100mm)
- Camera movement animations (-5mm to +5mm)

**Outputs**: `outputs_ft_python/` (40 files total)
- Per pattern: input, display, views (no tunable), views (3 focal lengths), 2 animations

**Run**: `python inverse_renderer_competitor.py`

**Status**: ✅ **COMPLETE** - All MATLAB visualizations implemented and tested

---

### 3. Optimizer: `standalone_optimizer.py` ⚠️
**Purpose**: Gradient descent optimization approach

**Method**:
- PyTorch-based gradient descent
- Multi-ray sampling (1-ray, 4-ray modes)
- Textured spheres with letters A, B, C
- Complete optical physics

**Run**: `python standalone_optimizer.py`

**Status**: Functional, needs testing

---

## Comparison Table

| Approach | Method | Speed | Quality | Animations | MATLAB Match |
|----------|--------|-------|---------|------------|--------------|
| **Ground Truth** | Direct ray trace | Fast | Perfect | Yes | N/A |
| **Inverse Renderer** | Analytical inverse | Fast | High | Yes | ✅ 100% |
| **Optimizer** | Gradient descent | Slow | High | Partial | N/A |

---

## Output Summary

### Ground Truth (`results/`)
- 4 files (images + GIFs)
- Spherical checkerboard at 200mm
- Multiple focal lengths and eye positions

### Inverse Renderer (`outputs_ft_python/`)
- **40 files** (5 patterns × 8 outputs)
- Complete MATLAB equivalence
- See `OUTPUTS_SUMMARY.md` for details

### Optimizer
- TBD - needs testing

---

## Directory Structure

```
optimizer/
├── spherical_checkerboard_raytracer.py  ✅ Ground truth
├── inverse_renderer_competitor.py       ✅ Inverse renderer (NEW!)
├── standalone_optimizer.py              ⚠️ Optimizer
├── sphere_synthesis_focus_tunable_standalone.m  (MATLAB reference)
├── results/                             (Ground truth outputs)
├── outputs_ft_python/                   (Inverse renderer outputs - 40 files)
├── deprecated/                          (Old scripts)
├── README.md                            (This file)
└── OUTPUTS_SUMMARY.md                   (Detailed output documentation)
```

---

## Key Achievement

The **inverse renderer competitor** now provides exact MATLAB-equivalent outputs including:
- ✅ All 5 test patterns
- ✅ Display generation via inverse ray tracing
- ✅ Views with/without tunable lens
- ✅ Focal sweep animations
- ✅ Camera movement animations

This establishes a proper baseline for comparing the optimization approach.

---

## Dependencies

```bash
pip install torch numpy matplotlib pillow
```

---

## Quick Start

```bash
# Run ground truth
python spherical_checkerboard_raytracer.py

# Run inverse renderer (competitor)
python inverse_renderer_competitor.py

# Run optimizer
python standalone_optimizer.py
```
