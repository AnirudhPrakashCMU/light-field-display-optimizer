#!/usr/bin/env python3
"""
Parallel Checkerboard Optimizer
Optimizes multiple checkerboards simultaneously for better GPU utilization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from standalone_optimizer import (
    LightFieldDisplay, create_spherical_checkerboard,
    render_eye_view_target, render_eye_view_through_display,
    device, samples_per_pixel_override
)
import os
from datetime import datetime
import matplotlib.pyplot as plt

def optimize_parallel_batch(checkerboard_configs, iterations, resolution, local_results_dir, batch_size=4):
    """
    Optimize multiple checkerboards in parallel batches

    Args:
        checkerboard_configs: List of (square_size, actual_squares) tuples
        iterations: Number of optimization iterations
        resolution: Rendering resolution
        local_results_dir: Directory to save results
        batch_size: Number of checkerboards to optimize in parallel (default 4)
    """
    print(f"\n🚀 PARALLEL BATCH OPTIMIZATION")
    print(f"   Optimizing {len(checkerboard_configs)} checkerboards in batches of {batch_size}")
    print(f"   {iterations} iterations, {resolution}x{resolution} resolution")

    eye_position = torch.tensor([0.0, 0.0, 0.0], device=device)
    eye_focal_length = 30.0

    # Process in batches
    all_results = []

    for batch_start in range(0, len(checkerboard_configs), batch_size):
        batch_end = min(batch_start + batch_size, len(checkerboard_configs))
        batch_configs = checkerboard_configs[batch_start:batch_end]

        print(f"\n{'='*70}")
        print(f"🔥 BATCH {batch_start//batch_size + 1}: Optimizing {len(batch_configs)} checkerboards in parallel")
        print(f"{'='*70}")

        # Create display systems for this batch
        display_systems = []
        optimizers = []
        targets = []
        scene_names = []

        for square_size, actual_squares in batch_configs:
            # Create display system
            display_system = LightFieldDisplay(resolution=1024, num_planes=10)
            optimizer = optim.AdamW(display_system.parameters(), lr=0.03)

            # Generate target
            scene_objects = create_spherical_checkerboard(square_size)
            with torch.no_grad():
                target = render_eye_view_target(eye_position, eye_focal_length, scene_objects, resolution)

            display_systems.append(display_system)
            optimizers.append(optimizer)
            targets.append(target)
            scene_names.append(f"checkerboard_{actual_squares}x{actual_squares}")

            print(f"   ✓ Initialized {actual_squares}x{actual_squares}")

        # Check GPU memory
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024**3
            print(f"   GPU Memory: {memory_used:.2f} GB")

        # Optimize all in parallel
        print(f"\n   Starting parallel optimization...")
        start_time = datetime.now()
        loss_histories = [[] for _ in range(len(batch_configs))]

        for iteration in range(iterations):
            # Process all display systems in this batch
            for idx, (display_system, optimizer, target) in enumerate(zip(display_systems, optimizers, targets)):
                optimizer.zero_grad()

                # Render through display
                simulated_image = render_eye_view_through_display(
                    eye_position, eye_focal_length, display_system, resolution
                )

                # Compute loss
                loss = torch.mean((simulated_image - target) ** 2)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(display_system.parameters(), max_norm=1.0)
                optimizer.step()

                with torch.no_grad():
                    display_system.display_images.clamp_(0, 1)

                loss_histories[idx].append(loss.item())

            # Print progress
            if iteration % 10 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                memory_used = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
                avg_loss = sum([h[-1] for h in loss_histories]) / len(loss_histories)
                print(f"     Iter {iteration}: Avg Loss = {avg_loss:.6f}, GPU = {memory_used:.2f} GB, Time = {elapsed:.1f}s")

        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"\n   ✅ Batch complete in {elapsed:.1f}s")

        # Save results for this batch
        for idx, (square_size, actual_squares) in enumerate(batch_configs):
            scene_name = scene_names[idx]
            scene_local_dir = f'{local_results_dir}/scenes/{scene_name}'
            os.makedirs(scene_local_dir, exist_ok=True)

            # Save loss curve
            plt.figure(figsize=(10, 6))
            plt.plot(loss_histories[idx], 'b-', linewidth=2)
            plt.title(f'{scene_name} - Loss Convergence')
            plt.xlabel('Iteration')
            plt.ylabel('MSE Loss')
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
            plt.savefig(f'{scene_local_dir}/loss_curve.png', dpi=150, bbox_inches='tight')
            plt.close()

            result = {
                'scene_name': scene_name,
                'final_loss': loss_histories[idx][-1],
                'initial_loss': loss_histories[idx][0],
                'display_system': display_systems[idx],
                'elapsed_time': elapsed / len(batch_configs)  # Approximate per-scene time
            }
            all_results.append(result)

            print(f"     {scene_name}: Loss {loss_histories[idx][-1]:.6f} → {loss_histories[idx][0]:.6f} ({(1-loss_histories[idx][-1]/loss_histories[idx][0])*100:.1f}% reduction)")

        # Clear GPU memory
        del display_systems, optimizers, targets
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return all_results

if __name__ == "__main__":
    print("🚀 PARALLEL CHECKERBOARD OPTIMIZATION")
    print("=" * 70)

    # Configure checkerboards
    checkerboard_configs = []
    for num_squares in range(25, 61, 5):
        square_size = 1000 // num_squares
        actual_squares = 1000 // square_size
        checkerboard_configs.append((square_size, actual_squares))

    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    local_results_dir = f'/workspace/parallel_optimization_{timestamp}'
    os.makedirs(local_results_dir, exist_ok=True)

    # Run parallel optimization
    results = optimize_parallel_batch(
        checkerboard_configs=checkerboard_configs,
        iterations=50,
        resolution=512,
        local_results_dir=local_results_dir,
        batch_size=4  # Optimize 4 checkerboards at once
    )

    print(f"\n{'='*70}")
    print(f"✅ ALL OPTIMIZATIONS COMPLETE")
    print(f"   Results saved to: {local_results_dir}")
    print(f"{'='*70}")
