# A100 Optimization Summary (40GB VRAM)

## Changes Made for A100

### 1. **Display Resolution**
- **Before**: 512x512 displays
- **After**: 1024x1024 displays
- **Memory**: ~96 MB for 8 display planes (3 channels × 1024² × 8 planes × 4 bytes)

### 2. **Rendering Resolution**
- **Before**: 128x128
- **After**: 512x512
- **Impact**: 16x more pixels per render

### 3. **Batch Sizes**
- **Before**: 512 pixels/batch
- **After**: 8192 pixels/batch
- **Impact**: 16x larger batches → better GPU utilization

### 4. **Optimization Iterations**
- **Before**: 50 iterations
- **After**: 100 iterations
- **Reason**: Higher LR (0.03) enables faster convergence

### 5. **Learning Rate**
- **Before**: 0.02
- **After**: 0.03
- **Reason**: Larger batches support higher learning rate

### 6. **Pre-generation Strategy**
- All 18 ground truth targets generated upfront
- Keeps targets in GPU memory
- Eliminates redundant ray tracing

### 7. **Fast Optimization Function**
- `optimize_single_scene_fast()` - minimal output generation
- Only saves loss curves (no progress GIFs during training)
- Focuses computation on optimization, not visualization

## Memory Usage Breakdown

### Per Checkerboard Optimization:
```
Display parameters:    96 MB  (8 × 3 × 1024² × 4 bytes)
Gradients:            96 MB  (same size as parameters)
Target image:          3 MB  (512² × 3 × 4 bytes)
Simulated image:       3 MB  (512² × 3 × 4 bytes)
Intermediate tensors: ~50 MB (ray tracing buffers)
-----------------------------------
Total per iteration: ~250 MB
```

### Total for 18 Checkerboards:
```
All targets stored:    54 MB  (18 × 3 MB)
Peak single opt:      250 MB
Expected total:      ~500 MB active, rest for CUDA ops
```

## Expected Performance

### Before (128x128, 50 iter):
- ~2-3 minutes per checkerboard
- ~40-50 minutes total

### After (512x512, 100 iter) on A100:
- ~5-8 minutes per checkerboard
- ~90-140 minutes total
- **Much higher quality** (16x resolution)

## VRAM Utilization

**Expected**: 2-5 GB out of 40 GB

**Why not 40GB?**
- Checkerboard optimization is compute-bound, not memory-bound
- Ray tracing kernels have sequential dependencies
- Display parameters are only 96 MB each
- Most VRAM available but not needed for this workload

**To use more VRAM, we could:**
1. Increase display resolution to 2048x2048 (would use ~400MB per display)
2. Increase rendering resolution to 1024x1024
3. Batch multiple checkerboards simultaneously
4. Increase rays per pixel from 1 to 4-16

## Verification

The optimization is still **100% REAL**:
- ✅ Real backpropagation (`loss.backward()`)
- ✅ Real gradient descent (`optimizer.step()`)
- ✅ Loss reduction tracked and printed
- ✅ GPU memory usage monitored every 10 iterations
- ✅ All 8 displays optimized independently

## Key Optimizations Applied

1. **Pre-computation**: All targets generated once
2. **Batching**: 16x larger batches (512 → 8192)
3. **Resolution**: 16x more pixels (128² → 512²)
4. **Display Quality**: 4x higher res (512² → 1024²)
5. **Convergence**: 2x iterations + higher LR
6. **Output**: Minimal visualization during training

## Files Modified
- `standalone_optimizer.py`:
  - Line 29: `samples_per_pixel_override = 1`
  - Line 267: `batch_size = min(8192, N)`
  - Line 353: `batch_size = min(8192, N)`
  - Line 479: `LightFieldDisplay(resolution=1024)`
  - Line 1183-1254: New `optimize_single_scene_fast()`
  - Line 1256-1340: Updated `run_checkerboard_density_sweep()`

## Run Command
```bash
python standalone_optimizer.py
```

## Expected Output
- 18 checkerboards optimized (26x26 to 62x62)
- Loss curves for each
- Final GIF of optimized eye views
- ZIP archive with all results
