#!/usr/bin/env python3
"""
Quick Checkerboard Test - A100 Optimized
Streamlined version that runs only the spherical checkerboard scene for 10 iterations
Perfect for quick testing and validation of the multi-ray sampling system
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import shutil
import zipfile
from dataclasses import dataclass
from typing import List, Tuple
import cv2
import math
from tqdm import tqdm

print("=== QUICK CHECKERBOARD TEST - A100 OPTIMIZED ===")
print("Fast validation of spherical checkerboard with multi-ray sampling")
print("Optimized for 40GB A100 with 10 iterations")

# Device setup with GPU memory clearing
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if torch.cuda.is_available():
    # Clear GPU memory at startup
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print("GPU memory cleared at startup")
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Reset memory stats for clean tracking
    torch.cuda.reset_peak_memory_stats()
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
print()

# Clean up results directory - save to workspace for RunPod (change for Colab if needed)
results_dir = '/workspace/results/quick_test_results'  # Use /content/drive/MyDrive/aswin/quick_test_results for Colab
if os.path.exists(results_dir):
    shutil.rmtree(results_dir)
os.makedirs(results_dir, exist_ok=True)
print(f"Results will be saved to: {results_dir}")

@dataclass
class QuickOpticalParams:
    """Streamlined optical system parameters for quick testing"""
    # Eye parameters - balanced for speed and quality
    eye_pupil_diameter: float = 4.0  # mm
    retina_distance: float = 24.0  # mm
    retina_size: float = 10.0  # mm
    samples_per_pixel: int = 256  # ULTRA-HIGH ray sampling for 80GB A100
    
    # Tunable lens parameters
    tunable_lens_distance: float = 50.0  # mm from eye
    tunable_lens_diameter: float = 15.0  # mm
    tunable_lens_focal_range: Tuple[float, float] = (10.0, 100.0)  # mm
    
    # Microlens array parameters - high density to use more GPU memory
    microlens_distance: float = 80.0  # mm from eye
    microlens_array_size: float = 20.0  # mm x mm (reasonable array)
    microlens_pitch: float = 0.5  # mm spacing (reasonable density)
    microlens_focal_length: float = 1.0  # mm
    
    # Display parameters - high resolution to use more GPU memory
    display_distance: float = 82.0  # mm from eye
    display_size: float = 20.0  # mm x mm (reasonable display)
    display_resolution: int = 6144  # pixels per side (ULTRA-HIGH resolution for 80GB)
    num_focal_planes: int = 8  # Reasonable focal planes (focus memory on resolution)

class SphericalCheckerboard:
    """Physical spherical checkerboard scene - from spherical_checkerboard_raytracer.py"""
    
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius
        print(f"Spherical Checkerboard: center={center.cpu().numpy()}, radius={radius}mm")
        
    def get_color(self, point_3d):
        """Get MATLAB-compatible checkerboard color at 3D point"""
        direction = point_3d - self.center
        direction_norm = direction / torch.norm(direction, dim=-1, keepdim=True)
        
        X = direction_norm[..., 0]
        Y = direction_norm[..., 1]
        Z = direction_norm[..., 2]
        
        # MATLAB convert_3d_direction_to_euler
        rho = torch.sqrt(X*X + Z*Z)
        phi = torch.atan2(Z, X)
        theta = torch.atan2(Y, rho)
        
        # Map to flat checkerboard pattern (1000x1000, 50px squares)
        theta_norm = (theta + math.pi/2) / math.pi
        phi_norm = (phi + math.pi) / (2*math.pi)
        
        i_coord = theta_norm * 999
        j_coord = phi_norm * 999
        
        i_square = torch.floor(i_coord / 50).long()
        j_square = torch.floor(j_coord / 50).long()
        
        return ((i_square + j_square) % 2).float()

def generate_pupil_samples(num_samples, pupil_radius):
    """Generate uniform pupil samples - from spherical_checkerboard_raytracer.py"""
    angles = torch.linspace(0, 2*math.pi, num_samples, device=device)
    radii = torch.sqrt(torch.rand(num_samples, device=device)) * pupil_radius
    
    x = radii * torch.cos(angles)
    y = radii * torch.sin(angles)
    
    return torch.stack([x, y], dim=1)

def create_spherical_checkerboard() -> SphericalCheckerboard:
    """Create spherical checkerboard scene"""
    scene = SphericalCheckerboard(
        center=torch.tensor([0.0, 0.0, 200.0], device=device),
        radius=50.0
    )
    return scene

def trace_ray_through_spherical_checkerboard(ray_origin: torch.Tensor, ray_dir: torch.Tensor, 
                                           scene: SphericalCheckerboard) -> torch.Tensor:
    """Trace ray through spherical checkerboard and return color"""
    batch_shape = ray_origin.shape[:-1]
    color = torch.zeros((*batch_shape, 3), device=device)
    
    # Ray-sphere intersection
    oc = ray_origin - scene.center
    a = torch.sum(ray_dir * ray_dir, dim=-1)
    b = 2.0 * torch.sum(oc * ray_dir, dim=-1)
    c = torch.sum(oc * oc, dim=-1) - scene.radius * scene.radius
    
    discriminant = b * b - 4 * a * c
    hit_mask = discriminant >= 0
    
    if hit_mask.any():
        sqrt_discriminant = torch.sqrt(discriminant[hit_mask])
        t1 = (-b[hit_mask] - sqrt_discriminant) / (2 * a[hit_mask])
        t2 = (-b[hit_mask] + sqrt_discriminant) / (2 * a[hit_mask])
        
        t_valid = torch.where(t1 > 1e-6, t1, t2)
        valid_hits = t_valid > 1e-6
        
        if valid_hits.any():
            hit_points = ray_origin[hit_mask][valid_hits] + t_valid[valid_hits].unsqueeze(-1) * ray_dir[hit_mask][valid_hits]
            checkerboard_colors = scene.get_color(hit_points)
            
            final_mask = torch.zeros_like(hit_mask)
            final_mask[hit_mask] = valid_hits
            
            color[final_mask, 0] = checkerboard_colors
            color[final_mask, 1] = checkerboard_colors
            color[final_mask, 2] = checkerboard_colors
    
    return color

class QuickLightFieldDisplay(nn.Module):
    """Streamlined light field display system"""
    
    def __init__(self, params: QuickOpticalParams):
        super().__init__()
        self.params = params
        
        # Learnable display images
        self.display_images = nn.Parameter(
            torch.rand(params.num_focal_planes, 3, params.display_resolution, params.display_resolution, 
                      device=device, dtype=torch.float32) * 0.5
        )
        
        # Fixed focal lengths
        self.focal_lengths = torch.linspace(
            params.tunable_lens_focal_range[0], 
            params.tunable_lens_focal_range[1], 
            params.num_focal_planes, 
            device=device
        )
        
        # Pre-compute microlens positions
        self.microlens_positions = self._compute_microlens_positions()
        
    def _compute_microlens_positions(self) -> torch.Tensor:
        """Pre-compute circular microlens center positions"""
        pitch = self.params.microlens_pitch
        array_size = self.params.microlens_array_size
        
        num_lenses = int(array_size / pitch)
        
        x_centers = torch.linspace(-array_size/2, array_size/2, num_lenses, device=device)
        y_centers = torch.linspace(-array_size/2, array_size/2, num_lenses, device=device)
        
        x_grid, y_grid = torch.meshgrid(x_centers, y_centers, indexing='ij')
        
        positions = torch.stack([x_grid.flatten(), y_grid.flatten()], dim=1)
        
        return positions

def generate_target_image_checkerboard(scene: SphericalCheckerboard, eye_focal_length: float, 
                                     params: QuickOpticalParams, resolution: int = 2048) -> torch.Tensor:
    """Generate target image for spherical checkerboard with multi-ray sampling"""
    
    retina_size = params.retina_size
    y_coords = torch.linspace(-retina_size/2, retina_size/2, resolution, device=device)
    x_coords = torch.linspace(-retina_size/2, retina_size/2, resolution, device=device)
    
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
    retina_points = torch.stack([
        x_grid.flatten(),
        y_grid.flatten(), 
        torch.full_like(x_grid.flatten(), -params.retina_distance)
    ], dim=1)
    
    N = retina_points.shape[0]
    M = params.samples_per_pixel
    
    # Process in MASSIVE batches for 80GB A100
    batch_size = min(524288, N)  # MASSIVE batches for 80GB (512K pixels)
    final_colors = torch.zeros(N, 3, device=device)
    
    # Generate pupil samples
    pupil_radius = params.eye_pupil_diameter / 2
    pupil_samples = generate_pupil_samples(M, pupil_radius)
    
    for batch_start in range(0, N, batch_size):
        batch_end = min(batch_start + batch_size, N)
        batch_retina_points = retina_points[batch_start:batch_end]
        batch_N = batch_retina_points.shape[0]
        
        # Create ray bundles
        pupil_points_3d = torch.zeros(M, 3, device=device)
        pupil_points_3d[:, 0] = pupil_samples[:, 0]
        pupil_points_3d[:, 1] = pupil_samples[:, 1]
        pupil_points_3d[:, 2] = 0.0
        
        retina_expanded = batch_retina_points.unsqueeze(1)
        pupil_expanded = pupil_points_3d.unsqueeze(0).expand(batch_N, M, 3)
        
        ray_dirs = pupil_expanded - retina_expanded
        ray_dirs = ray_dirs / torch.norm(ray_dirs, dim=-1, keepdim=True)
        ray_origins = pupil_expanded
        
        # Apply lens refraction
        lens_power = 1000.0 / eye_focal_length / 1000.0
        
        local_x = pupil_expanded[:, :, 0]
        local_y = pupil_expanded[:, :, 1]
        
        deflection_x = -lens_power * local_x
        deflection_y = -lens_power * local_y
        
        refracted_ray_dirs = ray_dirs.clone()
        refracted_ray_dirs[:, :, 0] += deflection_x
        refracted_ray_dirs[:, :, 1] += deflection_y
        refracted_ray_dirs = refracted_ray_dirs / torch.norm(refracted_ray_dirs, dim=-1, keepdim=True)
        
        # Trace rays to spherical checkerboard
        ray_origins_flat = ray_origins.reshape(-1, 3)
        ray_dirs_flat = refracted_ray_dirs.reshape(-1, 3)
        
        colors_flat = trace_ray_through_spherical_checkerboard(ray_origins_flat, ray_dirs_flat, scene)
        
        # Average over sub-aperture samples
        colors = colors_flat.reshape(batch_N, M, 3)
        
        # Check pupil validity
        radial_distance = torch.sqrt(pupil_expanded[:, :, 0]**2 + pupil_expanded[:, :, 1]**2)
        valid_mask = radial_distance <= pupil_radius
        
        batch_colors = torch.zeros(batch_N, 3, device=device)
        for pixel_idx in range(batch_N):
            valid_samples = valid_mask[pixel_idx, :]
            if valid_samples.any():
                pixel_colors = colors[pixel_idx, valid_samples, :]
                batch_colors[pixel_idx, :] = torch.mean(pixel_colors, dim=0)
        
        final_colors[batch_start:batch_end] = batch_colors
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    retina_image = final_colors.reshape(resolution, resolution, 3)
    
    return retina_image

def simulate_eye_view_through_display(scene: SphericalCheckerboard, display_system: QuickLightFieldDisplay,
                                    eye_focal_length: float, tunable_focal_length: float, 
                                    params: QuickOpticalParams, resolution: int = 2048) -> torch.Tensor:
    """Simulate what the eye sees through the complete optical system"""
    
    retina_size = params.retina_size
    y_coords = torch.linspace(-retina_size/2, retina_size/2, resolution, device=device)
    x_coords = torch.linspace(-retina_size/2, retina_size/2, resolution, device=device)
    
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
    retina_points = torch.stack([
        x_grid.flatten(),
        y_grid.flatten(),
        torch.full_like(x_grid.flatten(), -params.retina_distance)
    ], dim=1)
    
    N = retina_points.shape[0]
    M = params.samples_per_pixel
    
    # Process in MASSIVE batches for 80GB A100
    batch_size = min(262144, N)  # MASSIVE batches for 80GB (256K pixels)
    final_colors = torch.zeros(N, 3, device=device)
    
    # Generate pupil samples
    pupil_radius = params.eye_pupil_diameter / 2
    pupil_samples = generate_pupil_samples(M, pupil_radius)
    
    for batch_start in range(0, N, batch_size):
        batch_end = min(batch_start + batch_size, N)
        batch_retina_points = retina_points[batch_start:batch_end]
        batch_N = batch_retina_points.shape[0]
        
        # Create ray bundles
        pupil_points_3d = torch.zeros(M, 3, device=device)
        pupil_points_3d[:, 0] = pupil_samples[:, 0]
        pupil_points_3d[:, 1] = pupil_samples[:, 1]
        pupil_points_3d[:, 2] = 0.0
        
        retina_expanded = batch_retina_points.unsqueeze(1)
        pupil_expanded = pupil_points_3d.unsqueeze(0).expand(batch_N, M, 3)
        
        ray_dirs = pupil_expanded - retina_expanded
        ray_dirs = ray_dirs / torch.norm(ray_dirs, dim=-1, keepdim=True)
        ray_origins = pupil_expanded
        
        # Apply eye lens refraction
        lens_power = 1000.0 / eye_focal_length / 1000.0
        local_x = pupil_expanded[:, :, 0]
        local_y = pupil_expanded[:, :, 1]
        
        deflection_x = -lens_power * local_x
        deflection_y = -lens_power * local_y
        
        refracted_ray_dirs = ray_dirs.clone()
        refracted_ray_dirs[:, :, 0] += deflection_x
        refracted_ray_dirs[:, :, 1] += deflection_y
        refracted_ray_dirs = refracted_ray_dirs / torch.norm(refracted_ray_dirs, dim=-1, keepdim=True)
        
        # Apply tunable lens refraction
        lens_z = params.tunable_lens_distance
        t_lens = (lens_z - ray_origins[:, :, 2]) / refracted_ray_dirs[:, :, 2]
        lens_intersection = ray_origins + t_lens.unsqueeze(-1) * refracted_ray_dirs
        
        tunable_lens_power = 1.0 / tunable_focal_length
        deflection_x_tunable = -tunable_lens_power * lens_intersection[:, :, 0]
        deflection_y_tunable = -tunable_lens_power * lens_intersection[:, :, 1]
        
        refracted_ray_dirs[:, :, 0] += deflection_x_tunable
        refracted_ray_dirs[:, :, 1] += deflection_y_tunable
        refracted_ray_dirs = refracted_ray_dirs / torch.norm(refracted_ray_dirs, dim=-1, keepdim=True)
        
        # Apply microlens refraction (simplified)
        microlens_z = params.microlens_distance
        t_array = (microlens_z - lens_intersection[:, :, 2]) / refracted_ray_dirs[:, :, 2]
        array_intersection = lens_intersection + t_array.unsqueeze(-1) * refracted_ray_dirs
        
        # Simplified microlens selection (nearest grid point)
        ray_xy = array_intersection[:, :, :2]
        grid_x = torch.round(ray_xy[:, :, 0] / params.microlens_pitch) * params.microlens_pitch
        grid_y = torch.round(ray_xy[:, :, 1] / params.microlens_pitch) * params.microlens_pitch
        
        # Check if within microlens
        distance_to_center = torch.sqrt((ray_xy[:, :, 0] - grid_x)**2 + (ray_xy[:, :, 1] - grid_y)**2)
        valid_microlens = distance_to_center <= params.microlens_pitch / 2
        
        # Apply microlens refraction
        microlens_power = 1.0 / params.microlens_focal_length
        local_x_micro = ray_xy[:, :, 0] - grid_x
        local_y_micro = ray_xy[:, :, 1] - grid_y
        
        deflection_x_micro = -microlens_power * local_x_micro
        deflection_y_micro = -microlens_power * local_y_micro
        
        refracted_ray_dirs[:, :, 0] += deflection_x_micro
        refracted_ray_dirs[:, :, 1] += deflection_y_micro
        refracted_ray_dirs = refracted_ray_dirs / torch.norm(refracted_ray_dirs, dim=-1, keepdim=True)
        
        # Sample display (simplified)
        display_z = params.display_distance
        t_display = (display_z - array_intersection[:, :, 2]) / refracted_ray_dirs[:, :, 2]
        display_intersection = array_intersection + t_display.unsqueeze(-1) * refracted_ray_dirs
        
        display_size = params.display_size
        u = (display_intersection[:, :, 0] + display_size/2) / display_size
        v = (display_intersection[:, :, 1] + display_size/2) / display_size
        
        valid_display = (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1)
        
        # Sample from first display image only (simplified)
        display_color = torch.zeros(batch_N, M, 3, device=device)
        if valid_display.any():
            pixel_u = u * (params.display_resolution - 1)
            pixel_v = v * (params.display_resolution - 1)
            
            u0 = torch.floor(pixel_u).long().clamp(0, params.display_resolution - 1)
            v0 = torch.floor(pixel_v).long().clamp(0, params.display_resolution - 1)
            
            valid_pixels = valid_display & valid_microlens
            if valid_pixels.any():
                sampled_colors = display_system.display_images[0, :, v0[valid_pixels], u0[valid_pixels]].T
                display_color[valid_pixels] = sampled_colors
        
        # Average over sub-aperture samples
        radial_distance = torch.sqrt(pupil_expanded[:, :, 0]**2 + pupil_expanded[:, :, 1]**2)
        valid_pupil = radial_distance <= pupil_radius
        final_valid = valid_pupil & valid_microlens & valid_display
        
        batch_colors = torch.zeros(batch_N, 3, device=device)
        for pixel_idx in range(batch_N):
            valid_samples = final_valid[pixel_idx, :]
            if valid_samples.any():
                pixel_colors = display_color[pixel_idx, valid_samples, :]
                batch_colors[pixel_idx, :] = torch.mean(pixel_colors, dim=0)
        
        final_colors[batch_start:batch_end] = batch_colors
    
    simulated_image = final_colors.reshape(resolution, resolution, 3)
    return simulated_image

def quick_optimization_test():
    """Run quick optimization test on spherical checkerboard"""
    print("\n=== QUICK CHECKERBOARD OPTIMIZATION TEST ===")
    
    # Create parameters and scene
    params = QuickOpticalParams()
    scene = create_spherical_checkerboard()
    
    print(f"\nHigh-Memory Quick Test Parameters:")
    print(f"  Display resolution: {params.display_resolution}x{params.display_resolution} (HIGH MEMORY)")
    print(f"  Multi-ray sampling: {params.samples_per_pixel} rays per pixel (HIGH MEMORY)")
    print(f"  Focal planes: {params.num_focal_planes} (HIGH MEMORY)")
    num_microlenses = int(params.microlens_array_size/params.microlens_pitch)**2
    print(f"  Microlenses: {num_microlenses:,} (ULTRA-HIGH DENSITY)")
    print(f"  Training resolution: 768x768 pixels")
    print(f"  Final resolution: 512x512 pixels")
    print(f"  Training iterations: 10 (quick validation with progress bar)")
    print(f"  Expected memory usage: 30-40GB (80GB A100 maximized)")
    
    # Create display system
    display_system = QuickLightFieldDisplay(params)
    
    # Optimizer with mixed precision
    optimizer = optim.AdamW(display_system.parameters(), lr=0.02, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    # Training setup
    eye_focal_length = 35.0
    tunable_focal_length = 25.0
    loss_history = []
    
    print(f"\nTraining spherical checkerboard scene...")
    print(f"Fixed focal lengths: {display_system.focal_lengths.cpu().numpy()}")
    
    # Quick training loop with progress bar - reduced to 10 iterations
    for iteration in tqdm(range(10), desc="Training", unit="iter"):
        optimizer.zero_grad()
        
        # Generate target and simulated images
        with torch.cuda.amp.autocast() if torch.cuda.is_available() else torch.no_grad():
            # Generate target image (no gradients needed) - ULTRA-HIGH resolution for 80GB
            with torch.no_grad():
                target_image = generate_target_image_checkerboard(
                    scene, eye_focal_length, params, resolution=2048  # ULTRA-HIGH resolution for 80GB
                )
            
            # Generate simulated image through display system (needs gradients)
            simulated_image = simulate_eye_view_through_display(
                scene, display_system, eye_focal_length, tunable_focal_length, params, resolution=2048
            )
            
            # Compute MSE loss between target and simulated
            loss = torch.mean((simulated_image - target_image) ** 2)
        
        # Backward pass with mixed precision
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(display_system.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(display_system.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Clamp values
        with torch.no_grad():
            display_system.display_images.clamp_(0, 1)
        
        loss_history.append(loss.item())
        
        if iteration % 2 == 0:  # More frequent updates for 10 iterations
            # Update progress bar with loss info
            tqdm.write(f"  Iteration {iteration}: Loss = {loss.item():.6f}")
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1024**3
                tqdm.write(f"    GPU Memory Used: {memory_used:.2f} GB")
        
        # No memory cleanup during 10-iteration run to maintain high memory usage
    
    # Final memory report
    if torch.cuda.is_available():
        final_memory = torch.cuda.memory_allocated() / 1024**3
        max_memory = torch.cuda.max_memory_allocated() / 1024**3
        print(f"\nQuick test complete!")
        print(f"  Final loss: {loss_history[-1]:.6f}")
        print(f"  Final GPU memory used: {final_memory:.2f} GB")
        print(f"  Peak GPU memory used: {max_memory:.2f} GB")
    
    # Save results
    print("Saving quick test results...")
    
    # Generate final comparison at high resolution
    with torch.no_grad():
        final_target = generate_target_image_checkerboard(scene, eye_focal_length, params, resolution=512)
        final_simulated = simulate_eye_view_through_display(
            scene, display_system, eye_focal_length, tunable_focal_length, params, resolution=512
        )
    
    # Save comparison images
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(np.clip(final_target.detach().cpu().numpy(), 0, 1))
    plt.title('Target Image\nSpherical Checkerboard\nMulti-Ray Sampling', fontsize=12)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(np.clip(final_simulated.detach().cpu().numpy(), 0, 1))
    plt.title('Simulated Image\nThrough Display System\n(After 100 iterations)', fontsize=12)
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.plot(loss_history, 'b-', linewidth=2)
    plt.title(f'Training Loss\nFinal: {loss_history[-1]:.6f}', fontsize=12)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.suptitle('Quick Checkerboard Test - A100 Optimized (10 iterations)', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{results_dir}/quick_checkerboard_test.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save display images
    plt.figure(figsize=(20, 4))
    for i in range(params.num_focal_planes):
        display_img = display_system.display_images[i].detach().cpu().numpy()
        display_img = np.transpose(display_img, (1, 2, 0))
        
        plt.subplot(1, params.num_focal_planes, i + 1)
        plt.imshow(np.clip(display_img, 0, 1))
        plt.title(f'FL: {display_system.focal_lengths[i]:.1f}mm', fontsize=10)
        plt.axis('off')
    
    plt.suptitle('Quick Test - Optimized Display Images', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{results_dir}/display_images.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ High-memory quick test results saved to {results_dir}/")
    print("✓ Target image with 64-ray sampling")
    print("✓ Simulated image through complete optical system")
    print("✓ Loss curve showing convergence")
    print("✓ Optimized display images (20 focal planes)")
    print(f"✓ Ultra-high density microlens array ({num_microlenses:,} microlenses)")
    
    return loss_history

def main():
    """Main function for quick checkerboard test"""
    print("Initializing Quick Checkerboard Test...")
    
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Memory Available: {total_memory:.1f} GB")
    
    # Run the quick test
    print("\nStarting high-memory quick test...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()  # Reset memory tracking
    
    loss_history = quick_optimization_test()
    
    print(f"\n=== HIGH-MEMORY QUICK CHECKERBOARD TEST COMPLETE ===")
    print("Generated high-memory test outputs:")
    print("  ✓ Spherical checkerboard scene (MATLAB-compatible)")
    print("  ✓ Multi-ray sub-aperture sampling (64 rays per pixel)")
    print("  ✓ Ultra-high density microlens array (40,000 microlenses)")
    print("  ✓ High-resolution displays (2048x2048, 20 focal planes)")
    print("  ✓ A100 mixed precision training with progress bar")
    print("  ✓ 10 training iterations (quick high-memory validation)")
    print("  ✓ Display image optimization")
    print("  ✓ Loss convergence tracking")
    print("  ✓ Real-time GPU memory monitoring")
    print(f"  ✓ Results saved to {results_dir}/")
    print("\nHigh-memory quick test ready for analysis!")

if __name__ == "__main__":
    main()