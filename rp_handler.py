"""
RunPod serverless handler for Light Field Display Optimizer
Maps JSON input to light field optimization functions and returns JSON results
"""

import runpod
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import shutil
import zipfile
import json
import math
from datetime import datetime
import io
import base64

# Import the light field optimizer modules
import sys
sys.path.append('/app')

# Import spherical checkerboard functions
exec(open('/app/spherical_checkerboard_raytracer.py').read())

def setup_gpu():
    """Setup GPU for light field optimization"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        torch.backends.cudnn.benchmark = True
    
    return device

def run_quick_checkerboard_optimization(iterations=10, resolution=256, rays_per_pixel=8):
    """Run quick checkerboard optimization"""
    
    device = setup_gpu()
    
    # Create spherical checkerboard scene
    scene = SphericalCheckerboard(
        center=torch.tensor([0.0, 0.0, 200.0], device=device),
        radius=50.0
    )
    
    # Simple display system
    class SimpleDisplay(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.display_images = torch.nn.Parameter(
                torch.rand(4, 3, resolution*2, resolution*2, device=device) * 0.5
            )
            self.focal_lengths = torch.linspace(20, 60, 4, device=device)
    
    display_system = SimpleDisplay()
    optimizer = torch.optim.AdamW(display_system.parameters(), lr=0.02)
    
    # Training loop
    loss_history = []
    
    for iteration in range(iterations):
        optimizer.zero_grad()
        
        # Simplified loss computation
        simulated = torch.nn.functional.interpolate(
            display_system.display_images[0].unsqueeze(0), 
            size=(resolution, resolution), mode='bilinear'
        ).squeeze(0).permute(1, 2, 0)
        
        # Generate simple target
        target = generate_simple_checkerboard_target(scene, device, resolution)
        
        # Compute loss
        loss = torch.mean((simulated - target) ** 2)
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            display_system.display_images.clamp_(0, 1)
        
        loss_history.append(loss.item())
    
    return {
        'loss_history': loss_history,
        'final_loss': loss_history[-1],
        'iterations': iterations,
        'resolution': resolution,
        'rays_per_pixel': rays_per_pixel,
        'gpu_memory_used': torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    }

def generate_simple_checkerboard_target(scene, device, resolution):
    """Generate simple checkerboard target for serverless optimization"""
    
    retina_size = 10.0
    retina_distance = 24.0
    
    y_coords = torch.linspace(-retina_size/2, retina_size/2, resolution, device=device)
    x_coords = torch.linspace(-retina_size/2, retina_size/2, resolution, device=device)
    
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
    retina_points = torch.stack([
        x_grid.flatten(),
        y_grid.flatten(), 
        torch.full_like(x_grid.flatten(), -retina_distance)
    ], dim=1)
    
    # Simple ray tracing to sphere
    ray_dirs = scene.center - retina_points
    ray_dirs = ray_dirs / torch.norm(ray_dirs, dim=-1, keepdim=True)
    
    # Ray-sphere intersection
    oc = retina_points - scene.center
    a = torch.sum(ray_dirs * ray_dirs, dim=-1)
    b = 2.0 * torch.sum(oc * ray_dirs, dim=-1)
    c = torch.sum(oc * oc, dim=-1) - scene.radius * scene.radius
    
    discriminant = b * b - 4 * a * c
    hit_mask = discriminant >= 0
    
    colors = torch.zeros(retina_points.shape[0], 3, device=device)
    if hit_mask.any():
        sqrt_discriminant = torch.sqrt(discriminant[hit_mask])
        t = (-b[hit_mask] + sqrt_discriminant) / (2 * a[hit_mask])
        valid_hits = t > 1e-6
        
        if valid_hits.any():
            hit_points = retina_points[hit_mask][valid_hits] + t[valid_hits].unsqueeze(-1) * ray_dirs[hit_mask][valid_hits]
            checkerboard_colors = scene.get_color(hit_points)
            
            final_mask = torch.zeros_like(hit_mask)
            final_mask[hit_mask] = valid_hits
            
            colors[final_mask, :] = checkerboard_colors.unsqueeze(-1)
    
    return colors.reshape(resolution, resolution, 3)

def run_full_optimization(iterations=100, scenes=None, resolution=384):
    """Run full multi-scene optimization"""
    
    device = setup_gpu()
    
    if scenes is None:
        scenes = ['spherical_checkerboard']  # Default to checkerboard only for serverless
    
    results = {}
    
    for scene_name in scenes:
        if scene_name == 'spherical_checkerboard':
            scene_result = run_quick_checkerboard_optimization(iterations, resolution)
            results[scene_name] = scene_result
        else:
            # Add other scenes as needed
            results[scene_name] = {'status': 'scene_not_implemented'}
    
    return {
        'scenes_completed': list(results.keys()),
        'results': results,
        'total_scenes': len(scenes),
        'gpu_memory_peak': torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    }

def save_results_to_tmp(results, job_id="unknown"):
    """Save large results to /tmp and return paths"""
    
    output_dir = f"/tmp/light_field_results_{job_id}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results JSON
    results_path = f"{output_dir}/results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return {
        'results_path': results_path,
        'output_dir': output_dir,
        'status': 'results_saved'
    }

def run_task(inp: dict):
    """
    Main task runner - maps input parameters to light field optimization
    
    Input parameters:
    - task_type: 'quick_test' or 'full_optimization'
    - iterations: number of training iterations (default: 50)
    - resolution: target image resolution (default: 256)  
    - rays_per_pixel: multi-ray sampling density (default: 8)
    - scenes: list of scenes to optimize (default: ['spherical_checkerboard'])
    - save_large_outputs: whether to save detailed outputs (default: False)
    """
    
    task_type = inp.get("task_type", "quick_test")
    iterations = inp.get("iterations", 50)
    resolution = inp.get("resolution", 256)
    rays_per_pixel = inp.get("rays_per_pixel", 8)
    scenes = inp.get("scenes", ["spherical_checkerboard"])
    save_large_outputs = inp.get("save_large_outputs", False)
    job_id = inp.get("job_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    try:
        if task_type == "quick_test":
            # Quick checkerboard test
            results = run_quick_checkerboard_optimization(iterations, resolution, rays_per_pixel)
            
            if save_large_outputs:
                output_info = save_results_to_tmp(results, job_id)
                results.update(output_info)
            
            return {
                'status': 'success',
                'task_type': task_type,
                'results': results,
                'message': f'Quick test completed: {iterations} iterations, final loss: {results["final_loss"]:.6f}'
            }
            
        elif task_type == "full_optimization":
            # Full multi-scene optimization
            results = run_full_optimization(iterations, scenes, resolution)
            
            if save_large_outputs:
                output_info = save_results_to_tmp(results, job_id)
                results.update(output_info)
            
            return {
                'status': 'success', 
                'task_type': task_type,
                'results': results,
                'message': f'Full optimization completed: {len(scenes)} scenes, {iterations} iterations each'
            }
            
        else:
            return {
                'status': 'error',
                'message': f'Unknown task_type: {task_type}. Use "quick_test" or "full_optimization"'
            }
            
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Optimization failed: {str(e)}',
            'task_type': task_type,
            'input_params': inp
        }

def handler(job):
    """
    RunPod serverless handler entry point
    Receives: {"input": {...}}
    Returns: JSON-serializable result
    """
    
    inp = job.get("input", {}) or {}
    
    # Add GPU info to response
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            'gpu_name': torch.cuda.get_device_name(0),
            'gpu_memory_total': torch.cuda.get_device_properties(0).total_memory / 1e9,
            'gpu_memory_available': (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9
        }
    
    # Run the optimization task
    result = run_task(inp)
    
    # Add metadata
    result['gpu_info'] = gpu_info
    result['timestamp'] = datetime.now().isoformat()
    result['serverless_version'] = '1.0.0'
    
    return result

# Start the RunPod serverless worker
runpod.serverless.start({"handler": handler})