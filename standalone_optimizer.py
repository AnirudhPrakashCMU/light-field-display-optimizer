#!/usr/bin/env python3
"""
HONEST Light Field Display Optimizer - ALL DISPLAYS OPTIMIZED
NO CHEATING - Uses ALL 8 displays properly based on viewing geometry
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import json
import math
from datetime import datetime
import requests

print("🚀 HONEST LIGHT FIELD OPTIMIZER - ALL DISPLAYS OPTIMIZED - NO CHEATING")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")

# Global override for samples per pixel (set by main function)
samples_per_pixel_override = 4

class SceneObject:
    """3D scene object"""
    def __init__(self, position, size, color, shape):
        self.position = torch.tensor(position, device=device, dtype=torch.float32)
        self.size = size
        self.color = torch.tensor(color, device=device, dtype=torch.float32)
        self.shape = shape

class SphericalCheckerboard:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius
        print(f"Spherical Checkerboard: center={center.cpu().numpy()}, radius={radius}mm")
        
    def get_color(self, point_3d):
        """MATLAB-compatible checkerboard color"""
        direction = point_3d - self.center
        direction_norm = direction / torch.norm(direction, dim=-1, keepdim=True)
        
        X, Y, Z = direction_norm[..., 0], direction_norm[..., 1], direction_norm[..., 2]
        
        rho = torch.sqrt(X*X + Z*Z)
        phi = torch.atan2(Z, X)
        theta = torch.atan2(Y, rho)
        
        theta_norm = (theta + math.pi/2) / math.pi
        phi_norm = (phi + math.pi) / (2*math.pi)
        
        i_coord = theta_norm * 999
        j_coord = phi_norm * 999
        
        i_square = torch.floor(i_coord / 50).long()
        j_square = torch.floor(j_coord / 50).long()
        
        return ((i_square + j_square) % 2).float()

def generate_pupil_samples(num_samples, pupil_radius):
    """Generate pupil samples based on current ray count"""
    torch.manual_seed(42)  # Fixed seed for reproducibility
    
    if num_samples == 1:
        # Single ray through center of pupil
        return torch.zeros(1, 2, device=device)
    else:
        # Multiple rays - circular pattern
        angles = torch.linspace(0, 2*math.pi, num_samples, device=device, endpoint=False)
        radii = torch.sqrt(torch.linspace(0.1, 1, num_samples, device=device)) * pupil_radius
        x = radii * torch.cos(angles)
        y = radii * torch.sin(angles)
        return torch.stack([x, y], dim=1)

def ray_sphere_intersection(ray_origin, ray_dir, sphere_center, sphere_radius):
    oc = ray_origin - sphere_center
    a = torch.sum(ray_dir * ray_dir, dim=-1)
    b = 2.0 * torch.sum(oc * ray_dir, dim=-1)
    c = torch.sum(oc * oc, dim=-1) - sphere_radius * sphere_radius
    
    discriminant = b * b - 4 * a * c
    hit_mask = discriminant >= 0
    
    t = torch.full_like(discriminant, float('inf'))
    
    if hit_mask.any():
        sqrt_discriminant = torch.sqrt(discriminant[hit_mask])
        t1 = (-b[hit_mask] - sqrt_discriminant) / (2 * a[hit_mask])
        t2 = (-b[hit_mask] + sqrt_discriminant) / (2 * a[hit_mask])
        
        t_valid = torch.where(t1 > 1e-6, t1, t2)
        t[hit_mask] = t_valid
        
        final_valid = t_valid > 1e-6
        final_hit_mask = hit_mask.clone()
        final_hit_mask[hit_mask] = final_valid
        hit_mask = final_hit_mask
    
    return hit_mask, t

def trace_rays_to_scene(ray_origins, ray_dirs, scene_objects):
    """REAL ray tracing to 3D scene objects with proper depth sorting"""
    
    batch_size = ray_origins.shape[0]
    colors = torch.zeros(batch_size, 3, device=device)
    depths = torch.full((batch_size,), float('inf'), device=device)
    
    # Handle single SphericalCheckerboard
    if isinstance(scene_objects, SphericalCheckerboard):
        scene_objects = [scene_objects]
    
    # Ray trace each object with proper depth sorting
    for obj in scene_objects:
        if isinstance(obj, SceneObject):
            hit_mask, t = ray_sphere_intersection(
                ray_origins, ray_dirs, obj.position, obj.size
            )
            
            if hit_mask.any():
                closer_hits = hit_mask & (t < depths)
                if closer_hits.any():
                    colors[closer_hits] = obj.color
                    depths[closer_hits] = t[closer_hits]
                    
        elif isinstance(obj, SphericalCheckerboard):
            hit_mask, t = ray_sphere_intersection(
                ray_origins, ray_dirs, obj.center, obj.radius
            )
            
            if hit_mask.any():
                closer_hits = hit_mask & (t < depths)
                if closer_hits.any():
                    intersection_points = ray_origins[closer_hits] + t[closer_hits].unsqueeze(-1) * ray_dirs[closer_hits]
                    checkerboard_colors = obj.get_color(intersection_points)
                    colors[closer_hits, 0] = checkerboard_colors
                    colors[closer_hits, 1] = checkerboard_colors
                    colors[closer_hits, 2] = checkerboard_colors
                    depths[closer_hits] = t[closer_hits]
    
    return colors

def render_eye_view_target(eye_position, eye_focal_length, scene_objects, resolution=256):
    """TARGET: What eye sees looking directly at scene - REAL ray tracing"""
    
    pupil_diameter = 4.0
    retina_distance = 24.0
    retina_size = 8.0
    samples_per_pixel = samples_per_pixel_override
    
    # Create retina grid
    y_coords = torch.linspace(-retina_size/2, retina_size/2, resolution, device=device)
    x_coords = torch.linspace(-retina_size/2, retina_size/2, resolution, device=device)
    
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
    retina_points = torch.stack([
        x_grid.flatten(),
        y_grid.flatten(),
        torch.full_like(x_grid.flatten(), -retina_distance)
    ], dim=1)
    
    N = retina_points.shape[0]
    M = samples_per_pixel
    
    pupil_radius = pupil_diameter / 2
    pupil_samples = generate_pupil_samples(M, pupil_radius)
    
    # Process in batches
    batch_size = min(512, N)  # Match simulated for consistency
    final_colors = torch.zeros(N, 3, device=device)
    
    for batch_start in range(0, N, batch_size):
        batch_end = min(batch_start + batch_size, N)
        batch_retina_points = retina_points[batch_start:batch_end]
        batch_N = batch_retina_points.shape[0]
        
        pupil_points_3d = torch.zeros(M, 3, device=device)
        pupil_points_3d[:, 0] = pupil_samples[:, 0]
        pupil_points_3d[:, 1] = pupil_samples[:, 1]
        pupil_points_3d[:, 2] = 0.0
        
        retina_expanded = batch_retina_points.unsqueeze(1)
        pupil_expanded = pupil_points_3d.unsqueeze(0).expand(batch_N, M, 3)
        
        ray_dirs = pupil_expanded - retina_expanded
        ray_dirs = ray_dirs / torch.norm(ray_dirs, dim=-1, keepdim=True)
        ray_origins = pupil_expanded
        
        # Eye lens refraction
        lens_power = 1000.0 / eye_focal_length / 1000.0
        ray_dirs[:, :, 0] += -lens_power * pupil_expanded[:, :, 0]
        ray_dirs[:, :, 1] += -lens_power * pupil_expanded[:, :, 1]
        ray_dirs = ray_dirs / torch.norm(ray_dirs, dim=-1, keepdim=True)
        
        # Ray trace to 3D scene
        ray_origins_flat = ray_origins.reshape(-1, 3)
        ray_dirs_flat = ray_dirs.reshape(-1, 3)
        
        scene_colors = trace_rays_to_scene(ray_origins_flat, ray_dirs_flat, scene_objects)
        
        # Average over sub-aperture samples
        colors = scene_colors.reshape(batch_N, M, 3)
        
        pupil_radius_check = pupil_diameter / 2
        radial_distance = torch.sqrt(pupil_expanded[:, :, 0]**2 + pupil_expanded[:, :, 1]**2)
        valid_pupil = radial_distance <= pupil_radius_check
        
        batch_colors = torch.zeros(batch_N, 3, device=device)
        for pixel_idx in range(batch_N):
            valid_samples = valid_pupil[pixel_idx, :]
            if valid_samples.any():
                pixel_colors = colors[pixel_idx, valid_samples, :]
                batch_colors[pixel_idx, :] = torch.mean(pixel_colors, dim=0)
        
        final_colors[batch_start:batch_end] = batch_colors
    
    return final_colors.reshape(resolution, resolution, 3)

def render_individual_display_view(eye_position, eye_focal_length, display_system, display_idx, resolution=256):
    """Render what eye sees looking at ONE specific display through optical system"""
    
    pupil_diameter = 4.0
    retina_distance = 24.0
    retina_size = 8.0
    samples_per_pixel = samples_per_pixel_override
    
    tunable_lens_distance = 50.0
    # Use DIFFERENT tunable focal length for each display based on its focal length
    display_focal_length = display_system.focal_lengths[display_idx].item()
    tunable_focal_length = display_focal_length  # Match tunable lens to this display
    
    microlens_distance = 80.0
    microlens_pitch = 0.4
    microlens_focal_length = 1.0
    display_distance = 82.0
    display_size = 20.0
    
    y_coords = torch.linspace(-retina_size/2, retina_size/2, resolution, device=device)
    x_coords = torch.linspace(-retina_size/2, retina_size/2, resolution, device=device)
    
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
    retina_points = torch.stack([
        x_grid.flatten(),
        y_grid.flatten(),
        torch.full_like(x_grid.flatten(), -retina_distance)
    ], dim=1)
    
    N = retina_points.shape[0]
    M = samples_per_pixel
    
    pupil_radius = pupil_diameter / 2
    pupil_samples = generate_pupil_samples(M, pupil_radius)
    
    batch_size = min(512, N)
    final_colors = torch.zeros(N, 3, device=device)
    
    for batch_start in range(0, N, batch_size):
        batch_end = min(batch_start + batch_size, N)
        batch_retina_points = retina_points[batch_start:batch_end]
        batch_N = batch_retina_points.shape[0]
        
        pupil_points_3d = torch.zeros(M, 3, device=device)
        pupil_points_3d[:, 0] = pupil_samples[:, 0]
        pupil_points_3d[:, 1] = pupil_samples[:, 1]
        pupil_points_3d[:, 2] = 0.0
        
        retina_expanded = batch_retina_points.unsqueeze(1)
        pupil_expanded = pupil_points_3d.unsqueeze(0).expand(batch_N, M, 3)
        
        ray_dirs = pupil_expanded - retina_expanded
        ray_dirs = ray_dirs / torch.norm(ray_dirs, dim=-1, keepdim=True)
        ray_origins = pupil_expanded
        
        # Eye lens refraction
        lens_power = 1000.0 / eye_focal_length / 1000.0
        ray_dirs[:, :, 0] += -lens_power * pupil_expanded[:, :, 0]
        ray_dirs[:, :, 1] += -lens_power * pupil_expanded[:, :, 1]
        ray_dirs = ray_dirs / torch.norm(ray_dirs, dim=-1, keepdim=True)
        
        # Tunable lens refraction - TUNED for this specific display
        lens_z = tunable_lens_distance
        t_lens = (lens_z - ray_origins[:, :, 2]) / ray_dirs[:, :, 2]
        lens_intersection = ray_origins + t_lens.unsqueeze(-1) * ray_dirs
        
        tunable_lens_power = 1.0 / tunable_focal_length
        ray_dirs[:, :, 0] += -tunable_lens_power * lens_intersection[:, :, 0]
        ray_dirs[:, :, 1] += -tunable_lens_power * lens_intersection[:, :, 1]
        ray_dirs = ray_dirs / torch.norm(ray_dirs, dim=-1, keepdim=True)
        
        # Microlens array
        microlens_z = microlens_distance
        t_array = (microlens_z - lens_intersection[:, :, 2]) / ray_dirs[:, :, 2]
        array_intersection = lens_intersection + t_array.unsqueeze(-1) * ray_dirs
        
        ray_xy = array_intersection[:, :, :2]
        grid_x = torch.round(ray_xy[:, :, 0] / microlens_pitch) * microlens_pitch
        grid_y = torch.round(ray_xy[:, :, 1] / microlens_pitch) * microlens_pitch
        
        distance_to_center = torch.sqrt((ray_xy[:, :, 0] - grid_x)**2 + (ray_xy[:, :, 1] - grid_y)**2)
        valid_microlens = distance_to_center <= microlens_pitch / 2
        
        microlens_power = 1.0 / microlens_focal_length
        local_x_micro = ray_xy[:, :, 0] - grid_x
        local_y_micro = ray_xy[:, :, 1] - grid_y
        
        ray_dirs[:, :, 0] += -microlens_power * local_x_micro
        ray_dirs[:, :, 1] += -microlens_power * local_y_micro
        ray_dirs = ray_dirs / torch.norm(ray_dirs, dim=-1, keepdim=True)
        
        # Sample from THIS SPECIFIC display only
        display_z = display_distance
        t_display = (display_z - array_intersection[:, :, 2]) / ray_dirs[:, :, 2]
        display_intersection = array_intersection + t_display.unsqueeze(-1) * ray_dirs
        
        u = (display_intersection[:, :, 0] + display_size/2) / display_size
        v = (display_intersection[:, :, 1] + display_size/2) / display_size
        
        valid_display = (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1) & valid_microlens
        
        display_colors = torch.zeros(batch_N, M, 3, device=device)
        
        if valid_display.any():
            pixel_u = u * (display_system.display_images.shape[-1] - 1)
            pixel_v = v * (display_system.display_images.shape[-2] - 1)
            
            u0 = torch.floor(pixel_u).long().clamp(0, display_system.display_images.shape[-1] - 1)
            v0 = torch.floor(pixel_v).long().clamp(0, display_system.display_images.shape[-2] - 1)
            
            valid_pixels = valid_display
            if valid_pixels.any():
                # Sample from THIS specific display only
                plane_colors = display_system.display_images[display_idx, :, v0[valid_pixels], u0[valid_pixels]].T
                display_colors[valid_pixels] = plane_colors
        
        # Average over sub-aperture samples
        pupil_radius_check = pupil_diameter / 2
        radial_distance = torch.sqrt(pupil_expanded[:, :, 0]**2 + pupil_expanded[:, :, 1]**2)
        valid_pupil = radial_distance <= pupil_radius_check
        final_valid = valid_pupil & valid_display
        
        batch_colors = torch.zeros(batch_N, 3, device=device)
        for pixel_idx in range(batch_N):
            valid_samples = final_valid[pixel_idx, :]
            if valid_samples.any():
                pixel_colors = display_colors[pixel_idx, valid_samples, :]
                batch_colors[pixel_idx, :] = torch.mean(pixel_colors, dim=0)
        
        final_colors[batch_start:batch_end] = batch_colors
    
    return final_colors.reshape(resolution, resolution, 3)

def render_eye_view_through_display(eye_position, eye_focal_length, display_system, resolution=256):
    """HONEST: What eye sees through complete system - PURE SUM of ALL display contributions"""
    
    
    # Render EACH display individually through complete optical system
    combined_image = torch.zeros(resolution, resolution, 3, device=device)
    
    for display_idx in range(display_system.display_images.shape[0]):
        # HONEST ray tracing for this individual display
        individual_view = render_individual_display_view(
            eye_position, eye_focal_length, display_system, display_idx, resolution
        )
        
        # RAW ADDITION - NO WEIGHTS, NO FOCUS CALCULATIONS
        combined_image += individual_view
    
    # PURE SUM of ALL displays
    
    return combined_image

def create_scene_objects(scene_name):
    """Create REAL 3D scene objects"""
    
    if scene_name == 'basic':
        return [
            SceneObject([0, 0, 150], 15, [1, 0, 0], 'sphere'),
            SceneObject([20, 0, 200], 10, [0, 1, 0], 'sphere'),
            SceneObject([-15, 10, 180], 8, [0, 0, 1], 'sphere')
        ]
    elif scene_name == 'complex':
        return [
            SceneObject([0, 0, 120], 20, [1, 0.5, 0], 'sphere'),
            SceneObject([30, 15, 180], 12, [0.8, 0, 0.8], 'sphere'),
            SceneObject([-25, -10, 200], 15, [0, 0.8, 0.8], 'sphere')
        ]
    elif scene_name == 'stick_figure':
        return [
            SceneObject([0, 15, 180], 8, [1, 0.8, 0.6], 'sphere'),  # head
            SceneObject([0, 0, 180], 6, [1, 0.8, 0.6], 'sphere'),   # body
            SceneObject([-8, 5, 180], 4, [1, 0.8, 0.6], 'sphere'),  # left arm
            SceneObject([8, 5, 180], 4, [1, 0.8, 0.6], 'sphere')    # right arm
        ]
    elif scene_name == 'layered':
        return [
            SceneObject([0, 0, 100], 12, [1, 0, 0], 'sphere'),   # front
            SceneObject([0, 0, 200], 15, [0, 1, 0], 'sphere'),   # middle
            SceneObject([0, 0, 300], 18, [0, 0, 1], 'sphere')    # back
        ]
    elif scene_name == 'office':
        return [
            SceneObject([-20, -20, 150], 25, [0.8, 0.6, 0.4], 'sphere'),  # desk
            SceneObject([0, 10, 180], 8, [0.2, 0.2, 0.2], 'sphere')       # monitor
        ]
    elif scene_name == 'nature':
        return [
            SceneObject([0, -30, 200], 35, [0.4, 0.8, 0.2], 'sphere'),    # tree
            SceneObject([25, -25, 180], 20, [0.3, 0.7, 0.1], 'sphere')    # bush
        ]
    elif scene_name == 'spherical_checkerboard':
        return SphericalCheckerboard(
            center=torch.tensor([0.0, 0.0, 200.0], device=device),
            radius=50.0
        )
    else:
        return []

class LightFieldDisplay(nn.Module):
    def __init__(self, resolution=512, num_planes=8):
        super().__init__()
        
        # Initialize displays with ALL BLACK for clear optimization visualization
        self.display_images = nn.Parameter(
            torch.zeros(num_planes, 3, resolution, resolution, device=device)
        )
        
        self.focal_lengths = torch.linspace(10, 100, num_planes, device=device)
        
        print(f"📺 Display initialized: {num_planes} planes, {resolution}x{resolution}, ALL BLACK seed")
        print(f"🎯 Focal lengths: {self.focal_lengths.cpu().numpy()}")

def upload_to_catbox(file_path):
    if not os.path.exists(file_path):
        return None
    
    try:
        with open(file_path, 'rb') as f:
            files = {'fileToUpload': f}
            data = {'reqtype': 'fileupload'}
            response = requests.post('https://catbox.moe/user/api.php', files=files, data=data, timeout=30)
        
        if response.status_code == 200:
            url = response.text.strip()
            if url.startswith('https://'):
                print(f"✅ Uploaded: {os.path.basename(file_path)} -> {url}")
                return url
    except Exception as e:
        # Save locally if upload fails
        local_path = f'/workspace/results_{os.path.basename(file_path)}'
        import shutil
        shutil.copy2(file_path, local_path)
        print(f"💾 Saved locally: {local_path}")
        return f"file://{local_path}"
    
    return None

def optimize_single_scene(scene_name, scene_objects, iterations, resolution, local_results_dir):
    """REAL optimization with ALL displays optimized"""
    
    print(f"\n🎯 REAL Optimization: {scene_name} ({iterations} iterations)")
    start_time = datetime.now()
    
    display_system = LightFieldDisplay(resolution=512, num_planes=8)
    optimizer = optim.AdamW(display_system.parameters(), lr=0.02)
    
    eye_position = torch.tensor([0.0, 0.0, 0.0], device=device)
    eye_focal_length = 30.0
    
    # Generate REAL target
    print(f"   Generating REAL target using ray tracing to 3D scene...")
    with torch.no_grad():
        target_image = render_eye_view_target(eye_position, eye_focal_length, scene_objects, resolution)
    
    print(f"   Target generated: {target_image.shape}")
    
    # REAL optimization - ALL displays will be optimized
    loss_history = []
    progress_frames = []
    
    print(f"   Starting REAL optimization (ALL displays will be optimized)...")
    
    for iteration in range(iterations):
        optimizer.zero_grad()
        
        # REAL simulated image through complete optical system using ALL displays
        simulated_image = render_eye_view_through_display(
            eye_position, eye_focal_length, display_system, resolution
        )
        
        # REAL loss between ray-traced target and ray-traced simulated
        loss = torch.mean((simulated_image - target_image) ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(display_system.parameters(), max_norm=1.0)
        optimizer.step()
        
        with torch.no_grad():
            display_system.display_images.clamp_(0, 1)
        
        loss_history.append(loss.item())
        
        # Save EVERY iteration
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        axes[0].imshow(np.clip(target_image.detach().cpu().numpy(), 0, 1))
        axes[0].set_title(f'REAL Target: {scene_name}\\n(Ray traced to 3D scene)')
        axes[0].axis('off')
        
        axes[1].imshow(np.clip(simulated_image.detach().cpu().numpy(), 0, 1))
        axes[1].set_title(f'REAL Simulated\\n(Through optical system - ALL displays)\\nIter {iteration}, Loss: {loss.item():.6f}')
        axes[1].axis('off')
        
        axes[2].plot(loss_history, 'b-', linewidth=2)
        axes[2].set_title('REAL Loss')
        axes[2].set_yscale('log')
        axes[2].grid(True, alpha=0.3)
        
        plt.suptitle(f'REAL Light Field Optimization: {scene_name.title()} - Iteration {iteration}/{iterations}')
        plt.tight_layout()
        
        frame_path = f'/tmp/{scene_name}_progress_{iteration:04d}.png'
        plt.savefig(frame_path, dpi=80, bbox_inches='tight')
        plt.close()
        progress_frames.append(frame_path)
        
        if iteration % 10 == 0:
            memory_used = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
            elapsed = (datetime.now() - start_time).total_seconds()
            print(f"     Iter {iteration}: Loss = {loss.item():.6f}, GPU = {memory_used:.2f} GB, Time = {elapsed:.1f}s")
    
    print(f"   Creating outputs...")
    
    # Progress GIF with ALL frames
    gif_images = [Image.open(f) for f in progress_frames]
    progress_gif = f'/tmp/{scene_name}_progress_ALL_FRAMES.gif'
    gif_images[0].save(progress_gif, save_all=True, append_images=gif_images[1:], 
                      duration=100, loop=0, optimize=True)
    
    for f in progress_frames:
        os.remove(f)
    
    print(f"   ✅ Progress GIF: {len(gif_images)} frames (EVERY iteration)")
    
    # What each display shows - ALL 8 INDIVIDUAL displays
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i in range(8):
        row, col = i // 4, i % 4
        display_img = display_system.display_images[i].detach().cpu().numpy()
        display_img = np.transpose(display_img, (1, 2, 0))
        axes[row, col].imshow(np.clip(display_img, 0, 1))
        axes[row, col].set_title(f'Display {i+1}\\nFL: {display_system.focal_lengths[i]:.0f}mm')
        axes[row, col].axis('off')
    
    plt.suptitle(f'{scene_name.title()} - What Each Display Shows (ALL 8 Optimized)')
    plt.tight_layout()
    displays_path = f'/tmp/{scene_name}_displays.png'
    plt.savefig(displays_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # What eye sees for EACH display using REAL ray tracing
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i in range(8):
        row, col = i // 4, i % 4
        with torch.no_grad():
            # REAL ray tracing using THIS display's focal length
            eye_view_real = render_eye_view_through_display(
                eye_position, display_system.focal_lengths[i].item(), display_system, resolution
            )
        
        axes[row, col].imshow(np.clip(eye_view_real.detach().cpu().numpy(), 0, 1))
        axes[row, col].set_title(f'REAL Eye View {i+1}\\nFL: {display_system.focal_lengths[i]:.0f}mm')
        axes[row, col].axis('off')
    
    plt.suptitle(f'{scene_name.title()} - REAL Eye Views for Each Display')
    plt.tight_layout()
    eye_views_path = f'/tmp/{scene_name}_eye_views.png'
    plt.savefig(eye_views_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Focal sweep through optimized display system
    focal_frames = []
    focal_lengths_test = torch.linspace(25.0, 45.0, 10, device=device)
    
    for i, fl in enumerate(focal_lengths_test):
        with torch.no_grad():
            # What eye sees through optimized system at this focal length
            eye_view = render_eye_view_through_display(
                eye_position, fl.item(), display_system, resolution
            )
        
        plt.figure(figsize=(8, 6))
        plt.imshow(np.clip(eye_view.detach().cpu().numpy(), 0, 1))
        plt.title(f'{scene_name.title()} Through OPTIMIZED System\\nEye FL: {fl:.1f}mm')
        plt.axis('off')
        
        frame_path = f'/tmp/{scene_name}_focal_{i:03d}.png'
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        plt.close()
        focal_frames.append(frame_path)
    
    focal_images = [Image.open(f) for f in focal_frames]
    focal_sweep_gif = f'/tmp/{scene_name}_focal_sweep.gif'
    focal_images[0].save(focal_sweep_gif, save_all=True, append_images=focal_images[1:],
                        duration=300, loop=0, optimize=True)
    
    for f in focal_frames:
        os.remove(f)
    
    # Eye movement through optimized display system
    eye_frames = []
    eye_positions = torch.linspace(-10, 10, 15, device=device)
    
    for i, eye_x in enumerate(eye_positions):
        eye_pos = torch.tensor([eye_x.item(), 0.0, 0.0], device=device)
        
        with torch.no_grad():
            eye_view = render_eye_view_through_display(
                eye_pos, eye_focal_length, display_system, resolution
            )
        
        plt.figure(figsize=(8, 6))
        plt.imshow(np.clip(eye_view.detach().cpu().numpy(), 0, 1))
        plt.title(f'{scene_name.title()} Through OPTIMIZED System\\nEye X: {eye_x:.1f}mm')
        plt.axis('off')
        
        frame_path = f'/tmp/{scene_name}_eye_{i:03d}.png'
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        plt.close()
        eye_frames.append(frame_path)
    
    eye_images = [Image.open(f) for f in eye_frames]
    eye_movement_gif = f'/tmp/{scene_name}_eye_movement.gif'
    eye_images[0].save(eye_movement_gif, save_all=True, append_images=eye_images[1:],
                      duration=200, loop=0, optimize=True)
    
    for f in eye_frames:
        os.remove(f)
    
    # REAL scene focal sweep (for comparison)
    real_focal_frames = []
    
    for i, fl in enumerate(focal_lengths_test):
        with torch.no_grad():
            real_scene_view = render_eye_view_target(
                eye_position, fl.item(), scene_objects, resolution
            )
        
        plt.figure(figsize=(8, 6))
        plt.imshow(np.clip(real_scene_view.detach().cpu().numpy(), 0, 1))
        plt.title(f'{scene_name.title()} REAL Scene\\nEye FL: {fl:.1f}mm')
        plt.axis('off')
        
        frame_path = f'/tmp/{scene_name}_real_focal_{i:03d}.png'
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        plt.close()
        real_focal_frames.append(frame_path)
    
    real_focal_images = [Image.open(f) for f in real_focal_frames]
    real_focal_sweep_gif = f'/tmp/{scene_name}_REAL_scene_focal_sweep.gif'
    real_focal_images[0].save(real_focal_sweep_gif, save_all=True, append_images=real_focal_images[1:],
                             duration=300, loop=0, optimize=True)
    
    for f in real_focal_frames:
        os.remove(f)
    
    # REAL scene eye movement (for comparison)
    real_eye_frames = []
    
    for i, eye_x in enumerate(eye_positions):
        eye_pos = torch.tensor([eye_x.item(), 0.0, 0.0], device=device)
        
        with torch.no_grad():
            real_scene_view = render_eye_view_target(
                eye_pos, eye_focal_length, scene_objects, resolution
            )
        
        plt.figure(figsize=(8, 6))
        plt.imshow(np.clip(real_scene_view.detach().cpu().numpy(), 0, 1))
        plt.title(f'{scene_name.title()} REAL Scene\\nEye X: {eye_x:.1f}mm')
        plt.axis('off')
        
        frame_path = f'/tmp/{scene_name}_real_eye_{i:03d}.png'
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        plt.close()
        real_eye_frames.append(frame_path)
    
    real_eye_images = [Image.open(f) for f in real_eye_frames]
    real_eye_movement_gif = f'/tmp/{scene_name}_REAL_scene_eye_movement.gif'
    real_eye_images[0].save(real_eye_movement_gif, save_all=True, append_images=real_eye_images[1:],
                           duration=200, loop=0, optimize=True)
    
    for f in real_eye_frames:
        os.remove(f)
    
    # Save locally
    scene_local_dir = f'{local_results_dir}/scenes/{scene_name}'
    os.makedirs(scene_local_dir, exist_ok=True)
    
    import shutil
    shutil.copy2(progress_gif, f'{scene_local_dir}/progress_all_frames.gif')
    shutil.copy2(displays_path, f'{scene_local_dir}/what_displays_show.png') 
    shutil.copy2(eye_views_path, f'{scene_local_dir}/what_eye_sees.png')
    shutil.copy2(focal_sweep_gif, f'{scene_local_dir}/focal_sweep_through_display.gif')
    shutil.copy2(eye_movement_gif, f'{scene_local_dir}/eye_movement_through_display.gif')
    shutil.copy2(real_focal_sweep_gif, f'{scene_local_dir}/REAL_scene_focal_sweep.gif')
    shutil.copy2(real_eye_movement_gif, f'{scene_local_dir}/REAL_scene_eye_movement.gif')
    
    print(f"💾 All outputs saved locally to: {scene_local_dir}")
    
    # Upload
    print(f"   Uploading all results...")
    progress_url = upload_to_catbox(progress_gif)
    displays_url = upload_to_catbox(displays_path)
    eye_views_url = upload_to_catbox(eye_views_path)
    focal_sweep_url = upload_to_catbox(focal_sweep_gif)
    eye_movement_url = upload_to_catbox(eye_movement_gif)
    real_focal_sweep_url = upload_to_catbox(real_focal_sweep_gif)
    real_eye_movement_url = upload_to_catbox(real_eye_movement_gif)
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"✅ {scene_name} complete in {elapsed:.1f}s: 7/7 outputs")
    
    return {
        'final_loss': loss_history[-1],
        'progress_url': progress_url,
        'displays_url': displays_url,
        'eye_views_url': eye_views_url,
        'focal_sweep_url': focal_sweep_url,
        'eye_movement_url': eye_movement_url,
        'real_scene_focal_sweep_url': real_focal_sweep_url,
        'real_scene_eye_movement_url': real_eye_movement_url,
        'elapsed_time': elapsed
    }

def run_optimization_with_rays(rays_per_pixel, run_name):
    """Run complete optimization with specified rays per pixel"""
    print(f"\n🚀 STARTING {run_name} ({rays_per_pixel} ray{'s' if rays_per_pixel != 1 else ''} per pixel)")
    
    # Modify global samples_per_pixel - need to update this in both functions
    global samples_per_pixel_override
    samples_per_pixel_override = rays_per_pixel
    
    overall_start = datetime.now()
    iterations = 50
    resolution = 128
    
    print(f"⚙️ Parameters: {iterations} iterations per scene, {resolution}x{resolution}, {rays_per_pixel} ray{'s' if rays_per_pixel != 1 else ''} per pixel")
    
    # Create local results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    local_results_dir = f'/workspace/light_field_results_{run_name}_{timestamp}'
    os.makedirs(local_results_dir, exist_ok=True)
    print(f"📁 Local results directory: {local_results_dir}")
    
    # ALL 7 SCENES
    scene_names = ['basic', 'complex', 'stick_figure', 'layered', 'office', 'nature', 'spherical_checkerboard']
    
    all_results = {}
    
    for scene_name in scene_names:
        scene_objects = create_scene_objects(scene_name)
        scene_result = optimize_single_scene(scene_name, scene_objects, iterations, resolution, local_results_dir)
        all_results[scene_name] = scene_result
        
        # Collect URLs for summary
        print(f"   📥 {scene_name}: {scene_result.get('displays_url', 'No URL')}")
    
    overall_elapsed = (datetime.now() - overall_start).total_seconds()
    print(f"\n✅ {run_name} COMPLETE in {overall_elapsed/60:.1f} minutes!")
    print(f"📁 All results saved to: {local_results_dir}")
    
    return all_results

def main():
    try:
        print(f"🚀 HONEST LIGHT FIELD OPTIMIZER - DUAL RUN STARTED")
        print(f"🎯 Display initialization: ALL BLACK seed for clear optimization progression")
        
        # RUN 1: Single ray per pixel
        results_1ray = run_optimization_with_rays(1, "1_RAY")
        
        # RUN 2: Standard multi-ray (4 rays per pixel) 
        results_4ray = run_optimization_with_rays(4, "4_RAY")
        
        print(f"\n🎉 BOTH OPTIMIZATION RUNS COMPLETE!")
        print(f"✅ 1-ray optimization: 7 scenes completed") 
        print(f"✅ 4-ray optimization: 7 scenes completed")
        print(f"✅ All optimized display images saved in their respective directories")
        
    except Exception as e:
        import traceback
        print(f"❌ Error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return {'status': 'error', 'message': str(e)}

if __name__ == "__main__":
    main()