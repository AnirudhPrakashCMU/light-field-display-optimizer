#!/bin/bash

# Complete Light Field Optimizer Runner for RunPod Web Terminal
# Kills old sessions, sets up fresh environment, runs optimization in tmux

echo "🚀 COMPLETE LIGHT FIELD OPTIMIZER - TMUX RUNNER"
echo "="*60

# Kill any existing tmux sessions to start fresh
echo "🧹 Cleaning up old tmux sessions..."
tmux kill-server 2>/dev/null || echo "   No existing tmux sessions to kill"

# Set up environment variables
echo "⚙️ Setting up environment..."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

# Create timestamped output directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="/workspace/light_field_results_${TIMESTAMP}"
mkdir -p ${OUTPUT_DIR}

echo "📁 Results will be saved to: ${OUTPUT_DIR}"

# GPU info
echo "🖥️ GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader

# Change to workspace
cd /workspace

# Clean up any existing repository
if [ -d "light-field-display-optimizer" ]; then
    echo "🗑️ Removing old repository..."
    rm -rf light-field-display-optimizer
fi

# Clone fresh repository
echo "📥 Cloning latest repository..."
git clone https://github.com/AnirudhPrakashCMU/light-field-display-optimizer.git

if [ ! -d "light-field-display-optimizer" ]; then
    echo "❌ Failed to clone repository"
    exit 1
fi

cd light-field-display-optimizer

# Check latest commit
echo "📋 Repository status:"
git log --oneline -1

# Install dependencies
echo "📦 Installing Python dependencies..."
pip install --quiet torch torchvision matplotlib pillow requests numpy

# Create the optimization script
cat > complete_optimization.py << 'EOF'
#!/usr/bin/env python3

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

print("🚀 COMPLETE LIGHT FIELD OPTIMIZER - ALL 7 SCENES")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    initial_memory = torch.cuda.memory_allocated() / 1024**3
    print(f"   Initial memory: {initial_memory:.2f} GB")

class SphericalCheckerboard:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius
        print(f"Spherical Checkerboard: center={center.cpu().numpy()}, radius={radius}mm")
        
    def get_color(self, point_3d):
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
    angles = torch.linspace(0, 2*math.pi, num_samples, device=device)
    radii = torch.sqrt(torch.rand(num_samples, device=device)) * pupil_radius
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

def render_eye_view_target(eye_position, eye_focal_length, scene, resolution=256):
    pupil_diameter = 4.0
    retina_distance = 24.0
    retina_size = 8.0
    samples_per_pixel = 8
    
    eye_to_sphere = scene.center - eye_position
    eye_to_sphere_norm = eye_to_sphere / torch.norm(eye_to_sphere)
    forward_dir = eye_to_sphere_norm
    
    temp_up = torch.tensor([0.0, 0.0, 1.0], device=device)
    if torch.abs(torch.dot(forward_dir, temp_up)) > 0.9:
        temp_up = torch.tensor([1.0, 0.0, 0.0], device=device)
    
    right_dir = torch.cross(forward_dir, temp_up)
    right_dir = right_dir / torch.norm(right_dir)
    up_dir = torch.cross(right_dir, forward_dir)
    up_dir = up_dir / torch.norm(up_dir)
    
    u_coords = torch.linspace(-retina_size/2, retina_size/2, resolution, device=device)
    v_coords = torch.linspace(-retina_size/2, retina_size/2, resolution, device=device)
    u_grid, v_grid = torch.meshgrid(u_coords, v_coords, indexing='ij')
    
    retina_center = eye_position - retina_distance * forward_dir
    
    retina_points = (retina_center.unsqueeze(0).unsqueeze(0) + 
                    u_grid.unsqueeze(-1) * right_dir.unsqueeze(0).unsqueeze(0) +
                    v_grid.unsqueeze(-1) * up_dir.unsqueeze(0).unsqueeze(0))
    
    retina_points_flat = retina_points.reshape(-1, 3)
    N = retina_points_flat.shape[0]
    M = samples_per_pixel
    
    pupil_radius = pupil_diameter / 2
    pupil_samples = generate_pupil_samples(M, pupil_radius)
    
    pupil_points_3d = (eye_position.unsqueeze(0) + 
                      pupil_samples[:, 0:1] * right_dir.unsqueeze(0) +
                      pupil_samples[:, 1:2] * up_dir.unsqueeze(0))
    
    retina_expanded = retina_points_flat.unsqueeze(1)
    pupil_expanded = pupil_points_3d.unsqueeze(0)
    
    ray_dirs = pupil_expanded - retina_expanded
    ray_dirs = ray_dirs / torch.norm(ray_dirs, dim=-1, keepdim=True)
    ray_origins = pupil_expanded.expand(N, M, 3)
    
    lens_power = 1000.0 / eye_focal_length / 1000.0
    
    local_coords = pupil_expanded - eye_position.unsqueeze(0).unsqueeze(0)
    local_x = torch.sum(local_coords * right_dir, dim=-1).expand(N, M)
    local_y = torch.sum(local_coords * up_dir, dim=-1).expand(N, M)
    
    deflection_right = -lens_power * local_x
    deflection_up = -lens_power * local_y
    
    refracted_ray_dirs = ray_dirs.clone()
    refracted_ray_dirs += deflection_right.unsqueeze(-1) * right_dir.unsqueeze(0).unsqueeze(0)
    refracted_ray_dirs += deflection_up.unsqueeze(-1) * up_dir.unsqueeze(0).unsqueeze(0)
    refracted_ray_dirs = refracted_ray_dirs / torch.norm(refracted_ray_dirs, dim=-1, keepdim=True)
    
    ray_origins_flat = ray_origins.reshape(-1, 3)
    ray_dirs_flat = refracted_ray_dirs.reshape(-1, 3)
    
    hit_mask_flat, t_flat = ray_sphere_intersection(
        ray_origins_flat, ray_dirs_flat, scene.center, scene.radius
    )
    
    intersection_points_flat = ray_origins_flat + t_flat.unsqueeze(-1) * ray_dirs_flat
    colors_flat = torch.zeros_like(ray_origins_flat)
    
    if hit_mask_flat.any():
        valid_intersections = intersection_points_flat[hit_mask_flat]
        valid_colors = scene.get_color(valid_intersections)
        colors_flat[hit_mask_flat, 0] = valid_colors
        colors_flat[hit_mask_flat, 1] = valid_colors
        colors_flat[hit_mask_flat, 2] = valid_colors
    
    colors = colors_flat.reshape(N, M, 3)
    hit_mask = hit_mask_flat.reshape(N, M)
    
    final_colors = torch.zeros(N, 3, device=device)
    valid_hits_per_pixel = torch.sum(hit_mask, dim=1)
    
    pixels_with_hits = valid_hits_per_pixel > 0
    if pixels_with_hits.any():
        for pixel_idx in torch.where(pixels_with_hits)[0]:
            valid_samples = hit_mask[pixel_idx, :]
            if valid_samples.any():
                pixel_colors = colors[pixel_idx, valid_samples, :]
                final_colors[pixel_idx, :] = torch.mean(pixel_colors, dim=0)
    
    return final_colors.reshape(resolution, resolution, 3)

def render_eye_view_through_display(eye_position, eye_focal_length, display_system, resolution=256):
    pupil_diameter = 4.0
    retina_distance = 24.0
    retina_size = 8.0
    samples_per_pixel = 4
    
    tunable_lens_distance = 50.0
    tunable_focal_length = 25.0
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
        
        lens_power = 1000.0 / eye_focal_length / 1000.0
        ray_dirs[:, :, 0] += -lens_power * pupil_expanded[:, :, 0]
        ray_dirs[:, :, 1] += -lens_power * pupil_expanded[:, :, 1]
        ray_dirs = ray_dirs / torch.norm(ray_dirs, dim=-1, keepdim=True)
        
        lens_z = tunable_lens_distance
        t_lens = (lens_z - ray_origins[:, :, 2]) / ray_dirs[:, :, 2]
        lens_intersection = ray_origins + t_lens.unsqueeze(-1) * ray_dirs
        
        tunable_lens_power = 1.0 / tunable_focal_length
        ray_dirs[:, :, 0] += -tunable_lens_power * lens_intersection[:, :, 0]
        ray_dirs[:, :, 1] += -tunable_lens_power * lens_intersection[:, :, 1]
        ray_dirs = ray_dirs / torch.norm(ray_dirs, dim=-1, keepdim=True)
        
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
                sampled_colors = torch.zeros_like(display_colors[valid_pixels])
                for plane_idx in range(display_system.display_images.shape[0]):
                    plane_colors = display_system.display_images[plane_idx, :, v0[valid_pixels], u0[valid_pixels]].T
                    sampled_colors += plane_colors / display_system.display_images.shape[0]
                
                display_colors[valid_pixels] = sampled_colors
        
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

def generate_target_for_scene(scene_objects, eye_position, eye_focal_length, resolution):
    if isinstance(scene_objects, SphericalCheckerboard):
        return render_eye_view_target(eye_position, eye_focal_length, scene_objects, resolution)
    else:
        target = torch.zeros(resolution, resolution, 3, device=device)
        for obj in scene_objects:
            color = torch.tensor(obj['color'], device=device, dtype=torch.float32)
            y_coords = torch.linspace(-1, 1, resolution, device=device)
            x_coords = torch.linspace(-1, 1, resolution, device=device)
            y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
            
            center_x = float(obj['position'][0] / 100.0)
            center_y = float(obj['position'][1] / 100.0)
            radius_val = float(obj['size'] / 200.0)
            
            mask = ((x_grid - center_x)**2 + (y_grid - center_y)**2) < radius_val**2
            target[mask] = color.unsqueeze(0).unsqueeze(0)
        
        return target

class LightFieldDisplay(nn.Module):
    def __init__(self, resolution=512, num_planes=8):
        super().__init__()
        
        self.display_images = nn.Parameter(
            torch.rand(num_planes, 3, resolution, resolution, device=device) * 0.5
        )
        
        self.focal_lengths = torch.linspace(10, 100, num_planes, device=device)

def upload_to_catbox(file_path):
    if not os.path.exists(file_path):
        return None
    
    try:
        with open(file_path, 'rb') as f:
            files = {'fileToUpload': f}
            data = {'reqtype': 'fileupload'}
            response = requests.post('https://catbox.moe/user/api.php', files=files, data=data, timeout=120)
        
        if response.status_code == 200:
            url = response.text.strip()
            if url.startswith('https://'):
                print(f"✅ Uploaded: {os.path.basename(file_path)} -> {url}")
                return url
    except Exception as e:
        print(f"❌ Upload error for {os.path.basename(file_path)}: {e}")
    
    return None

def optimize_single_scene(scene_name, scene_objects, iterations, resolution):
    print(f"\n🎯 Optimizing {scene_name} ({iterations} iterations)")
    start_time = datetime.now()
    
    display_system = LightFieldDisplay(resolution=512, num_planes=8)
    optimizer = optim.AdamW(display_system.parameters(), lr=0.02)
    
    eye_position = torch.tensor([0.0, 0.0, 0.0], device=device)
    eye_focal_length = 30.0
    
    print(f"   Generating target...")
    with torch.no_grad():
        target_image = generate_target_for_scene(scene_objects, eye_position, eye_focal_length, resolution)
    
    loss_history = []
    progress_frames = []
    
    print(f"   Starting optimization...")
    
    for iteration in range(iterations):
        optimizer.zero_grad()
        
        simulated_image = render_eye_view_through_display(
            eye_position, eye_focal_length, display_system, resolution
        )
        
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
        axes[0].set_title(f'Target: {scene_name}')
        axes[0].axis('off')
        
        axes[1].imshow(np.clip(simulated_image.detach().cpu().numpy(), 0, 1))
        axes[1].set_title(f'Simulated\\nIter {iteration}, Loss: {loss.item():.6f}')
        axes[1].axis('off')
        
        axes[2].plot(loss_history, 'b-', linewidth=2)
        axes[2].set_title('Loss')
        axes[2].set_yscale('log')
        axes[2].grid(True, alpha=0.3)
        
        plt.suptitle(f'{scene_name.title()} - Iteration {iteration}/{iterations}')
        plt.tight_layout()
        
        frame_path = f'/tmp/{scene_name}_progress_{iteration:04d}.png'
        plt.savefig(frame_path, dpi=80, bbox_inches='tight')
        plt.close()
        progress_frames.append(frame_path)
        
        if iteration % 5 == 0:
            memory_used = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
            elapsed = (datetime.now() - start_time).total_seconds()
            print(f"     Iter {iteration}: Loss = {loss.item():.6f}, GPU = {memory_used:.2f} GB, Time = {elapsed:.1f}s")
    
    print(f"   Creating outputs...")
    
    # Progress GIF
    gif_images = [Image.open(f) for f in progress_frames]
    progress_gif = f'/tmp/{scene_name}_progress.gif'
    gif_images[0].save(progress_gif, save_all=True, append_images=gif_images[1:], 
                      duration=100, loop=0, optimize=True)
    
    for f in progress_frames:
        os.remove(f)
    
    # Display images
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i in range(8):
        row, col = i // 4, i % 4
        display_img = display_system.display_images[i].detach().cpu().numpy()
        display_img = np.transpose(display_img, (1, 2, 0))
        axes[row, col].imshow(np.clip(display_img, 0, 1))
        axes[row, col].set_title(f'Display {i+1}\\nFL: {display_system.focal_lengths[i]:.0f}mm')
        axes[row, col].axis('off')
    
    plt.suptitle(f'{scene_name.title()} - What Each Display Shows')
    plt.tight_layout()
    displays_path = f'/tmp/{scene_name}_displays.png'
    plt.savefig(displays_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Eye views
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i in range(8):
        row, col = i // 4, i % 4
        individual_display_view = torch.nn.functional.interpolate(
            display_system.display_images[i].unsqueeze(0), 
            size=(resolution, resolution), mode='bilinear'
        ).squeeze(0).permute(1, 2, 0)
        
        axes[row, col].imshow(np.clip(individual_display_view.detach().cpu().numpy(), 0, 1))
        axes[row, col].set_title(f'Eye View {i+1}\\nFL: {display_system.focal_lengths[i]:.0f}mm')
        axes[row, col].axis('off')
    
    plt.suptitle(f'{scene_name.title()} - What Eye Sees for Each Display')
    plt.tight_layout()
    eye_views_path = f'/tmp/{scene_name}_eye_views.png'
    plt.savefig(eye_views_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Focal sweep
    focal_frames = []
    focal_lengths_test = torch.linspace(25.0, 45.0, 10, device=device)
    
    for i, fl in enumerate(focal_lengths_test):
        with torch.no_grad():
            eye_view = render_eye_view_through_display(
                eye_position, fl.item(), display_system, resolution
            )
        
        plt.figure(figsize=(8, 6))
        plt.imshow(np.clip(eye_view.detach().cpu().numpy(), 0, 1))
        plt.title(f'{scene_name.title()} Through Optical System\\nEye FL: {fl:.1f}mm')
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
    
    # Eye movement
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
        plt.title(f'{scene_name.title()} Through Optical System\\nEye X: {eye_x:.1f}mm')
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
    
    # Upload all
    print(f"   Uploading results...")
    progress_url = upload_to_catbox(progress_gif)
    displays_url = upload_to_catbox(displays_path)
    eye_views_url = upload_to_catbox(eye_views_path)
    focal_sweep_url = upload_to_catbox(focal_sweep_gif)
    eye_movement_url = upload_to_catbox(eye_movement_gif)
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"✅ {scene_name} complete in {elapsed:.1f}s: 5/5 outputs uploaded")
    
    return {
        'final_loss': loss_history[-1],
        'progress_url': progress_url,
        'displays_url': displays_url,
        'eye_views_url': eye_views_url,
        'focal_sweep_url': focal_sweep_url,
        'eye_movement_url': eye_movement_url,
        'elapsed_time': elapsed
    }

def main():
    try:
        overall_start = datetime.now()
        print(f"🚀 COMPLETE LIGHT FIELD OPTIMIZER STARTED: {overall_start}")
        
        iterations = 25
        resolution = 128
        
        print(f"⚙️ Parameters: {iterations} iterations per scene, {resolution}x{resolution}")
        
        # ALL 7 SCENES
        all_scenes = {
            'basic': [
                {'position': [0, 0, 150], 'size': 15, 'color': [1, 0, 0], 'shape': 'sphere'},
                {'position': [20, 0, 200], 'size': 10, 'color': [0, 1, 0], 'shape': 'sphere'}
            ],
            'complex': [
                {'position': [0, 0, 120], 'size': 20, 'color': [1, 0.5, 0], 'shape': 'sphere'},
                {'position': [30, 15, 180], 'size': 12, 'color': [0.8, 0, 0.8], 'shape': 'sphere'}
            ],
            'stick_figure': [
                {'position': [0, 15, 180], 'size': 8, 'color': [1, 0.8, 0.6], 'shape': 'sphere'},
                {'position': [0, 0, 180], 'size': 6, 'color': [1, 0.8, 0.6], 'shape': 'sphere'}
            ],
            'layered': [
                {'position': [0, 0, 100], 'size': 12, 'color': [1, 0, 0], 'shape': 'sphere'},
                {'position': [0, 0, 200], 'size': 15, 'color': [0, 1, 0], 'shape': 'sphere'}
            ],
            'office': [
                {'position': [-20, -20, 150], 'size': 25, 'color': [0.8, 0.6, 0.4], 'shape': 'sphere'}
            ],
            'nature': [
                {'position': [0, -30, 200], 'size': 35, 'color': [0.4, 0.8, 0.2], 'shape': 'sphere'}
            ],
            'spherical_checkerboard': SphericalCheckerboard(
                center=torch.tensor([0.0, 0.0, 200.0], device=device),
                radius=50.0
            )
        }
        
        all_results = {}
        all_urls = {}
        
        for scene_name, scene_objects in all_scenes.items():
            scene_result = optimize_single_scene(scene_name, scene_objects, iterations, resolution)
            all_results[scene_name] = scene_result
            
            all_urls[f'{scene_name}_progress_gif'] = scene_result['progress_url']
            all_urls[f'{scene_name}_displays'] = scene_result['displays_url']
            all_urls[f'{scene_name}_eye_views'] = scene_result['eye_views_url']
            all_urls[f'{scene_name}_focal_sweep'] = scene_result['focal_sweep_url']
            all_urls[f'{scene_name}_eye_movement'] = scene_result['eye_movement_url']
            
            torch.cuda.empty_cache()
        
        overall_time = (datetime.now() - overall_start).total_seconds()
        
        print(f"\n" + "="*80)
        print("🎉 COMPLETE OPTIMIZATION FINISHED!")
        print(f"⏰ Total time: {overall_time:.1f} seconds ({overall_time/60:.1f} minutes)")
        print(f"📊 Scenes completed: {len(all_results)}")
        print(f"📥 Total download URLs: {len(all_urls)}")
        print("="*80)
        
        print(f"\n📥 ALL DOWNLOAD URLS:")
        for name, url in all_urls.items():
            print(f"   {name}: {url}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f'/workspace/complete_optimization_results_{timestamp}.json'
        
        complete_results = {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'total_time_seconds': overall_time,
            'scenes_completed': list(all_results.keys()),
            'total_scenes': len(all_results),
            'all_download_urls': all_urls,
            'scene_results': {name: {'final_loss': result['final_loss'], 'time': result['elapsed_time']} for name, result in all_results.items()},
            'optimization_specs': {
                'iterations_per_scene': iterations,
                'resolution': resolution,
                'total_scenes': 7,
                'outputs_per_scene': 5,
                'total_outputs': len(all_urls)
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(complete_results, f, indent=2)
        
        print(f"\n📋 Results saved to: {results_file}")
        print(f"✅ ALL 7 SCENES OPTIMIZATION COMPLETE!")
        
        return complete_results
        
    except Exception as e:
        import traceback
        print(f"❌ Error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return {'status': 'error', 'message': str(e)}

if __name__ == "__main__":
    main()
EOF

# Start tmux session and run optimization
echo ""
echo "🎯 Starting tmux session 'lightfield'..."
echo "📝 Instructions:"
echo "   1. Optimization will start automatically"
echo "   2. Press Ctrl+B then D to detach safely"
echo "   3. Close browser tab after detaching"
echo "   4. Reconnect later with: tmux attach -t lightfield"
echo ""
echo "⏰ Starting in 3 seconds..."
sleep 3

# Start tmux session with the optimization
tmux new-session -d -s lightfield -c /workspace/light-field-display-optimizer
tmux send-keys -t lightfield "python3 complete_optimization.py | tee ${OUTPUT_DIR}/optimization_log.txt" Enter

echo ""
echo "✅ Optimization started in tmux session 'lightfield'"
echo "📋 To attach and monitor: tmux attach -t lightfield"
echo "📋 To detach safely: Ctrl+B then D"
echo "📁 Results directory: ${OUTPUT_DIR}"
echo ""
echo "🎯 ATTACH TO TMUX SESSION NOW:"
echo "   tmux attach -t lightfield"
echo ""
echo "   Then press Ctrl+B, D to detach and close browser safely"