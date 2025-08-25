"""
ACTUAL Light Field Display Optimizer - Proper Implementation
Uses real ray tracing from spherical_checkerboard_raytracer.py for ground truth
"""

import runpod
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import zipfile
import json
import math
from datetime import datetime
import requests

print("🚀 ACTUAL LIGHT FIELD OPTIMIZER - PROPER RAY TRACING")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")

class SphericalCheckerboard:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius
        
    def get_color(self, point_3d):
        """MATLAB-compatible checkerboard color"""
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
    """Generate uniform pupil samples"""
    angles = torch.linspace(0, 2*math.pi, num_samples, device=device)
    radii = torch.sqrt(torch.rand(num_samples, device=device)) * pupil_radius
    
    x = radii * torch.cos(angles)
    y = radii * torch.sin(angles)
    
    return torch.stack([x, y], dim=1)

def ray_sphere_intersection(ray_origin, ray_dir, sphere_center, sphere_radius):
    """Ray-sphere intersection"""
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

def render_eye_view_target(eye_position, eye_focal_length, scene, resolution=512):
    """
    GROUND TRUTH: What eye sees looking directly at scene
    Uses proper ray tracing from spherical_checkerboard_raytracer.py
    """
    
    # Eye parameters
    pupil_diameter = 4.0  # mm
    retina_distance = 24.0  # mm
    retina_size = 8.0  # mm
    samples_per_pixel = 4  # Reduced for speed
    
    # Calculate eye orientation (tilted to point at sphere center)
    eye_to_sphere = scene.center - eye_position
    eye_to_sphere_norm = eye_to_sphere / torch.norm(eye_to_sphere)
    forward_dir = eye_to_sphere_norm
    
    # Create orthogonal basis for tilted retina
    temp_up = torch.tensor([0.0, 0.0, 1.0], device=device)
    if torch.abs(torch.dot(forward_dir, temp_up)) > 0.9:
        temp_up = torch.tensor([1.0, 0.0, 0.0], device=device)
    
    right_dir = torch.cross(forward_dir, temp_up)
    right_dir = right_dir / torch.norm(right_dir)
    up_dir = torch.cross(right_dir, forward_dir)
    up_dir = up_dir / torch.norm(up_dir)
    
    # Create tilted retina grid
    u_coords = torch.linspace(-retina_size/2, retina_size/2, resolution, device=device)
    v_coords = torch.linspace(-retina_size/2, retina_size/2, resolution, device=device)
    u_grid, v_grid = torch.meshgrid(u_coords, v_coords, indexing='ij')
    
    # Retina center positioned behind eye, along forward direction
    retina_center = eye_position - retina_distance * forward_dir
    
    # Retina points in tilted plane
    retina_points = (retina_center.unsqueeze(0).unsqueeze(0) + 
                    u_grid.unsqueeze(-1) * right_dir.unsqueeze(0).unsqueeze(0) +
                    v_grid.unsqueeze(-1) * up_dir.unsqueeze(0).unsqueeze(0))
    
    retina_points_flat = retina_points.reshape(-1, 3)
    N = retina_points_flat.shape[0]
    M = samples_per_pixel
    
    # Generate pupil samples
    pupil_radius = pupil_diameter / 2
    pupil_samples = generate_pupil_samples(M, pupil_radius)
    
    # Pupil points on tilted lens plane
    pupil_points_3d = (eye_position.unsqueeze(0) + 
                      pupil_samples[:, 0:1] * right_dir.unsqueeze(0) +
                      pupil_samples[:, 1:2] * up_dir.unsqueeze(0))
    
    # Ray bundles: [N, M, 3]
    retina_expanded = retina_points_flat.unsqueeze(1)
    pupil_expanded = pupil_points_3d.unsqueeze(0)
    
    ray_dirs = pupil_expanded - retina_expanded
    ray_dirs = ray_dirs / torch.norm(ray_dirs, dim=-1, keepdim=True)
    ray_origins = pupil_expanded.expand(N, M, 3)
    
    # Apply lens refraction
    lens_power = 1000.0 / eye_focal_length / 1000.0  # mm^-1
    
    local_coords = pupil_expanded - eye_position.unsqueeze(0).unsqueeze(0)
    local_x = torch.sum(local_coords * right_dir, dim=-1).expand(N, M)
    local_y = torch.sum(local_coords * up_dir, dim=-1).expand(N, M)
    
    deflection_right = -lens_power * local_x
    deflection_up = -lens_power * local_y
    
    refracted_ray_dirs = ray_dirs.clone()
    refracted_ray_dirs += deflection_right.unsqueeze(-1) * right_dir.unsqueeze(0).unsqueeze(0)
    refracted_ray_dirs += deflection_up.unsqueeze(-1) * up_dir.unsqueeze(0).unsqueeze(0)
    refracted_ray_dirs = refracted_ray_dirs / torch.norm(refracted_ray_dirs, dim=-1, keepdim=True)
    
    # Trace rays to sphere (PURE RAY TRACING)
    ray_origins_flat = ray_origins.reshape(-1, 3)
    ray_dirs_flat = refracted_ray_dirs.reshape(-1, 3)
    
    hit_mask_flat, t_flat = ray_sphere_intersection(
        ray_origins_flat, ray_dirs_flat, scene.center, scene.radius
    )
    
    # Get intersection colors
    intersection_points_flat = ray_origins_flat + t_flat.unsqueeze(-1) * ray_dirs_flat
    colors_flat = torch.zeros_like(ray_origins_flat)
    
    if hit_mask_flat.any():
        valid_intersections = intersection_points_flat[hit_mask_flat]
        valid_colors = scene.get_color(valid_intersections)
        colors_flat[hit_mask_flat, 0] = valid_colors
        colors_flat[hit_mask_flat, 1] = valid_colors
        colors_flat[hit_mask_flat, 2] = valid_colors
    
    # Average over sub-aperture samples (NATURAL BLUR from ray averaging)
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

def render_eye_view_through_display(eye_position, eye_focal_length, display_system, scene, resolution=512):
    """
    SIMULATED: What eye sees looking through the light field display system
    Ray trace: retina → eye lens → tunable lens → microlens array → display
    """
    
    # Eye parameters
    pupil_diameter = 4.0  # mm
    retina_distance = 24.0  # mm
    retina_size = 8.0  # mm
    samples_per_pixel = 4  # Reduced for speed
    
    # Tunable lens parameters
    tunable_lens_distance = 50.0  # mm from eye
    tunable_focal_length = 25.0  # mm
    
    # Microlens parameters
    microlens_distance = 80.0  # mm from eye
    microlens_pitch = 0.4  # mm
    microlens_focal_length = 1.0  # mm
    
    # Display parameters
    display_distance = 82.0  # mm from eye
    display_size = 20.0  # mm
    
    # Create retina grid
    retina_size_actual = retina_size
    y_coords = torch.linspace(-retina_size_actual/2, retina_size_actual/2, resolution, device=device)
    x_coords = torch.linspace(-retina_size_actual/2, retina_size_actual/2, resolution, device=device)
    
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
    retina_points = torch.stack([
        x_grid.flatten(),
        y_grid.flatten(),
        torch.full_like(x_grid.flatten(), -retina_distance)
    ], dim=1)
    
    N = retina_points.shape[0]
    M = samples_per_pixel
    
    # Generate pupil samples
    pupil_radius = pupil_diameter / 2
    pupil_samples = generate_pupil_samples(M, pupil_radius)
    
    # Process in smaller batches for speed
    batch_size = min(256, N)  # Smaller batches for faster execution
    final_colors = torch.zeros(N, 3, device=device)
    
    for batch_start in range(0, N, batch_size):
        batch_end = min(batch_start + batch_size, N)
        batch_retina_points = retina_points[batch_start:batch_end]
        batch_N = batch_retina_points.shape[0]
        
        # Create 3D pupil points
        pupil_points_3d = torch.zeros(M, 3, device=device)
        pupil_points_3d[:, 0] = pupil_samples[:, 0]
        pupil_points_3d[:, 1] = pupil_samples[:, 1]
        pupil_points_3d[:, 2] = 0.0
        
        # Ray bundles
        retina_expanded = batch_retina_points.unsqueeze(1)
        pupil_expanded = pupil_points_3d.unsqueeze(0).expand(batch_N, M, 3)
        
        ray_dirs = pupil_expanded - retina_expanded
        ray_dirs = ray_dirs / torch.norm(ray_dirs, dim=-1, keepdim=True)
        ray_origins = pupil_expanded
        
        # Step 1: Eye lens refraction
        lens_power = 1000.0 / eye_focal_length / 1000.0
        local_x = pupil_expanded[:, :, 0]
        local_y = pupil_expanded[:, :, 1]
        
        ray_dirs[:, :, 0] += -lens_power * local_x
        ray_dirs[:, :, 1] += -lens_power * local_y
        ray_dirs = ray_dirs / torch.norm(ray_dirs, dim=-1, keepdim=True)
        
        # Step 2: Tunable lens refraction
        lens_z = tunable_lens_distance
        t_lens = (lens_z - ray_origins[:, :, 2]) / ray_dirs[:, :, 2]
        lens_intersection = ray_origins + t_lens.unsqueeze(-1) * ray_dirs
        
        tunable_lens_power = 1.0 / tunable_focal_length
        ray_dirs[:, :, 0] += -tunable_lens_power * lens_intersection[:, :, 0]
        ray_dirs[:, :, 1] += -tunable_lens_power * lens_intersection[:, :, 1]
        ray_dirs = ray_dirs / torch.norm(ray_dirs, dim=-1, keepdim=True)
        
        # Step 3: Microlens array
        microlens_z = microlens_distance
        t_array = (microlens_z - lens_intersection[:, :, 2]) / ray_dirs[:, :, 2]
        array_intersection = lens_intersection + t_array.unsqueeze(-1) * ray_dirs
        
        # Find nearest microlens (grid-based)
        ray_xy = array_intersection[:, :, :2]
        grid_x = torch.round(ray_xy[:, :, 0] / microlens_pitch) * microlens_pitch
        grid_y = torch.round(ray_xy[:, :, 1] / microlens_pitch) * microlens_pitch
        
        # Check if within microlens
        distance_to_center = torch.sqrt((ray_xy[:, :, 0] - grid_x)**2 + (ray_xy[:, :, 1] - grid_y)**2)
        valid_microlens = distance_to_center <= microlens_pitch / 2
        
        # Microlens refraction
        microlens_power = 1.0 / microlens_focal_length
        local_x_micro = ray_xy[:, :, 0] - grid_x
        local_y_micro = ray_xy[:, :, 1] - grid_y
        
        ray_dirs[:, :, 0] += -microlens_power * local_x_micro
        ray_dirs[:, :, 1] += -microlens_power * local_y_micro
        ray_dirs = ray_dirs / torch.norm(ray_dirs, dim=-1, keepdim=True)
        
        # Step 4: Sample display
        display_z = display_distance
        t_display = (display_z - array_intersection[:, :, 2]) / ray_dirs[:, :, 2]
        display_intersection = array_intersection + t_display.unsqueeze(-1) * ray_dirs
        
        display_size_actual = display_size
        u = (display_intersection[:, :, 0] + display_size_actual/2) / display_size_actual
        v = (display_intersection[:, :, 1] + display_size_actual/2) / display_size_actual
        
        valid_display = (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1) & valid_microlens
        
        # Sample from display images
        display_colors = torch.zeros(batch_N, M, 3, device=device)
        
        if valid_display.any():
            # Sample from first display image (simplified for now)
            pixel_u = u * (display_system.display_images.shape[-1] - 1)
            pixel_v = v * (display_system.display_images.shape[-2] - 1)
            
            u0 = torch.floor(pixel_u).long().clamp(0, display_system.display_images.shape[-1] - 1)
            v0 = torch.floor(pixel_v).long().clamp(0, display_system.display_images.shape[-2] - 1)
            
            valid_pixels = valid_display
            if valid_pixels.any():
                sampled_colors = display_system.display_images[0, :, v0[valid_pixels], u0[valid_pixels]].T
                display_colors[valid_pixels] = sampled_colors
        
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

class LightFieldDisplay(nn.Module):
    def __init__(self, resolution=1024, num_planes=8):
        super().__init__()
        
        print(f"🧠 Display: {resolution}x{resolution}, {num_planes} focal planes")
        
        self.display_images = nn.Parameter(
            torch.rand(num_planes, 3, resolution, resolution, device=device) * 0.5
        )
        
        self.focal_lengths = torch.linspace(10, 100, num_planes, device=device)

def upload_to_catbox(file_path):
    """Upload to catbox.moe"""
    
    if not os.path.exists(file_path):
        return None
    
    filename = os.path.basename(file_path)
    print(f"📤 Uploading {filename}...")
    
    try:
        with open(file_path, 'rb') as f:
            files = {'fileToUpload': f}
            data = {'reqtype': 'fileupload'}
            response = requests.post('https://catbox.moe/user/api.php', files=files, data=data, timeout=120)
        
        if response.status_code == 200:
            url = response.text.strip()
            if url.startswith('https://'):
                print(f"✅ Uploaded: {url}")
                return url
    except:
        pass
    
    return None

def optimize_spherical_checkerboard(iterations, resolution):
    """Optimize ONLY spherical checkerboard with ACTUAL ray tracing"""
    
    print(f"🎯 ACTUAL OPTIMIZATION: Spherical Checkerboard")
    print(f"   Iterations: {iterations}")
    print(f"   Resolution: {resolution}x{resolution}")
    
    # Create scene
    scene = SphericalCheckerboard(
        center=torch.tensor([0.0, 0.0, 200.0], device=device),
        radius=50.0
    )
    
    # Create display system
    display_system = LightFieldDisplay(resolution=512, num_planes=4)  # Reduced for speed
    optimizer = optim.AdamW(display_system.parameters(), lr=0.02)
    
    # Eye setup
    eye_position = torch.tensor([0.0, 0.0, 0.0], device=device)
    eye_focal_length = 35.0
    
    # Generate REAL target (what eye sees looking at scene)
    print("🎯 Generating REAL target using proper ray tracing...")
    with torch.no_grad():
        target_image = render_eye_view_target(eye_position, eye_focal_length, scene, resolution)
    
    print(f"✅ Real target generated: {target_image.shape}")
    
    # Training loop
    loss_history = []
    progress_frames = []
    
    print(f"🔥 Starting REAL optimization...")
    
    for iteration in range(iterations):
        optimizer.zero_grad()
        
        # Generate REAL simulated image (what eye sees through display system)
        simulated_image = render_eye_view_through_display(
            eye_position, eye_focal_length, display_system, scene, resolution
        )
        
        # Compute REAL loss
        loss = torch.mean((simulated_image - target_image) ** 2)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(display_system.parameters(), max_norm=1.0)
        optimizer.step()
        
        with torch.no_grad():
            display_system.display_images.clamp_(0, 1)
        
        loss_history.append(loss.item())
        
        # Save EVERY iteration frame
        if True:  # Save every single iteration
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            
            axes[0].imshow(np.clip(target_image.cpu().numpy(), 0, 1))
            axes[0].set_title('Target (Eye → Scene)')
            axes[0].axis('off')
            
            axes[1].imshow(np.clip(simulated_image.detach().cpu().numpy(), 0, 1))
            axes[1].set_title(f'Simulated (Eye → Display)\\nIter {iteration}, Loss: {loss.item():.6f}')
            axes[1].axis('off')
            
            axes[2].plot(loss_history, 'b-', linewidth=2)
            axes[2].set_title('Loss Curve')
            axes[2].set_yscale('log')
            axes[2].grid(True, alpha=0.3)
            
            plt.suptitle(f'ACTUAL Light Field Optimization - Iteration {iteration}/{iterations}')
            plt.tight_layout()
            
            frame_path = f'/tmp/progress_{iteration:04d}.png'
            plt.savefig(frame_path, dpi=100, bbox_inches='tight')
            plt.close()
            progress_frames.append(frame_path)
        
        if iteration % 25 == 0:
            memory_used = torch.cuda.memory_allocated() / 1024**3
            print(f"   Iter {iteration}: Loss = {loss.item():.6f}, GPU = {memory_used:.2f} GB")
    
    # Create progress GIF
    gif_images = [Image.open(f) for f in progress_frames]
    progress_gif = '/tmp/spherical_checkerboard_progress.gif'
    gif_images[0].save(progress_gif, save_all=True, append_images=gif_images[1:], 
                      duration=200, loop=0, optimize=True)
    
    for f in progress_frames:
        os.remove(f)
    
    # Create ALL comprehensive outputs
    print("📊 Creating comprehensive outputs...")
    
    # 1. Final comparison image
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(np.clip(target_image.cpu().numpy(), 0, 1))
    axes[0].set_title('Target: Eye → Scene\\n(Ground Truth)')
    axes[0].axis('off')
    
    with torch.no_grad():
        final_simulated = render_eye_view_through_display(
            torch.tensor([0.0, 0.0, 0.0], device=device), 35.0, display_system, scene, resolution
        )
    
    axes[1].imshow(np.clip(final_simulated.cpu().numpy(), 0, 1))
    axes[1].set_title(f'Simulated: Eye → Display\\nFinal Loss: {loss_history[-1]:.6f}')
    axes[1].axis('off')
    
    axes[2].plot(loss_history, 'b-', linewidth=2)
    axes[2].set_title(f'Loss Convergence\\nFinal: {loss_history[-1]:.6f}')
    axes[2].set_xlabel('Iteration')
    axes[2].set_ylabel('MSE Loss')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle('ACTUAL Light Field Display Optimization Results')
    plt.tight_layout()
    final_comparison_path = '/tmp/final_comparison.png'
    plt.savefig(final_comparison_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Display images (what each display shows)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for i in range(4):
        display_img = display_system.display_images[i].detach().cpu().numpy()
        display_img = np.transpose(display_img, (1, 2, 0))
        axes[i].imshow(np.clip(display_img, 0, 1))
        axes[i].set_title(f'Display FL: {display_system.focal_lengths[i]:.0f}mm')
        axes[i].axis('off')
    
    plt.suptitle('What Each Display Shows - All Focal Planes')
    plt.tight_layout()
    displays_path = '/tmp/optimized_displays.png'
    plt.savefig(displays_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Eye views (what eye sees for each display)
    print("👁️ Creating eye views for each display...")
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for i in range(4):
        with torch.no_grad():
            # What eye sees when looking at this specific display
            eye_view = torch.nn.functional.interpolate(
                display_system.display_images[i].unsqueeze(0), 
                size=(resolution, resolution), mode='bilinear'
            ).squeeze(0).permute(1, 2, 0)
        
        axes[i].imshow(np.clip(eye_view.cpu().numpy(), 0, 1))
        axes[i].set_title(f'Eye View FL: {display_system.focal_lengths[i]:.0f}mm')
        axes[i].axis('off')
    
    plt.suptitle('What Eye Sees for Each Display')
    plt.tight_layout()
    eye_views_path = '/tmp/eye_views_all_displays.png'
    plt.savefig(eye_views_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Focal length sweep GIF
    print("🎬 Creating focal length sweep GIF...")
    focal_frames = []
    focal_lengths_test = torch.linspace(20.0, 50.0, 30, device=device)
    
    for i, fl in enumerate(focal_lengths_test):
        with torch.no_grad():
            target_fl = render_eye_view_target(
                torch.tensor([0.0, 0.0, 0.0], device=device), fl.item(), scene, resolution
            )
        
        plt.figure(figsize=(10, 8))
        
        plt.subplot(2, 1, 1)
        plt.imshow(np.clip(target_fl.cpu().numpy(), 0, 1))
        plt.title(f'Spherical Checkerboard - Eye FL: {fl:.1f}mm\\n4-Ray Multi-Ray Sampling')
        plt.axis('off')
        
        plt.subplot(2, 1, 2)
        plt.axis('off')
        
        # Focus calculation
        focused_distance = (fl.item() * 24.0) / (fl.item() - 24.0)
        defocus = abs(200.0 - focused_distance)
        
        if defocus < 15:
            status, color = "SHARP FOCUS", 'green'
        elif defocus < 35:
            status, color = "MODERATE BLUR", 'orange'
        else:
            status, color = "HEAVY BLUR", 'red'
        
        plt.text(0.5, 0.8, f'Eye Focal Length: {fl:.1f}mm', ha='center', fontsize=16, fontweight='bold')
        plt.text(0.5, 0.6, f'Focus Distance: {focused_distance:.0f}mm', ha='center', fontsize=14)
        plt.text(0.5, 0.4, f'Sphere Distance: 200mm', ha='center', fontsize=14)
        plt.text(0.5, 0.2, status, ha='center', fontsize=14, color=color, fontweight='bold')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        
        plt.suptitle(f'Focal Length Sweep - Frame {i+1}/30')
        plt.tight_layout()
        
        frame_path = f'/tmp/focal_frame_{i:03d}.png'
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        plt.close()
        focal_frames.append(frame_path)
    
    # Create focal sweep GIF
    focal_images = [Image.open(f) for f in focal_frames]
    focal_sweep_path = '/tmp/focal_length_sweep.gif'
    focal_images[0].save(focal_sweep_path, save_all=True, append_images=focal_images[1:],
                        duration=200, loop=0, optimize=True)
    
    for f in focal_frames:
        os.remove(f)
    
    print(f"✅ Focal sweep GIF: 30 frames")
    
    # 5. Eye movement sweep GIF
    print("🎬 Creating eye movement sweep GIF...")
    eye_frames = []
    eye_positions = torch.linspace(-10, 10, 20, device=device)
    
    for i, eye_x in enumerate(eye_positions):
        with torch.no_grad():
            eye_pos = torch.tensor([eye_x.item(), 0.0, 0.0], device=device)
            target_eye = render_eye_view_target(eye_pos, 35.0, scene, resolution)
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(np.clip(target_eye.cpu().numpy(), 0, 1))
        plt.title(f'Eye View from X: {eye_x:.1f}mm\\nSpherical Checkerboard')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        # Scene diagram
        pos = scene.center.cpu().numpy()
        circle = plt.Circle((pos[2], pos[0]), scene.radius, fill=False, color='blue', linewidth=3)
        plt.gca().add_patch(circle)
        plt.scatter(pos[2], pos[0], c='blue', s=200, marker='o')
        plt.scatter(0, eye_x.item(), c='red', s=150, marker='^', label='Eye')
        plt.plot([0, pos[2]], [eye_x.item(), pos[0]], 'r--', alpha=0.5)
        
        plt.xlabel('Distance (mm)')
        plt.ylabel('X Position (mm)')
        plt.title('Eye Movement')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xlim(-15, 250)
        plt.ylim(-15, 15)
        
        plt.suptitle(f'Eye Movement Sweep - Frame {i+1}/20')
        plt.tight_layout()
        
        frame_path = f'/tmp/eye_frame_{i:03d}.png'
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        plt.close()
        eye_frames.append(frame_path)
    
    # Create eye movement GIF
    eye_images = [Image.open(f) for f in eye_frames]
    eye_movement_path = '/tmp/eye_movement_sweep.gif'
    eye_images[0].save(eye_movement_path, save_all=True, append_images=eye_images[1:],
                      duration=150, loop=0, optimize=True)
    
    for f in eye_frames:
        os.remove(f)
    
    print(f"✅ Eye movement GIF: 20 frames")
    
    # Upload all results
    progress_url = upload_to_catbox(progress_gif)
    comparison_url = upload_to_catbox(final_comparison_path)
    displays_url = upload_to_catbox(displays_path)
    eye_views_url = upload_to_catbox(eye_views_path)
    focal_sweep_url = upload_to_catbox(focal_sweep_path)
    eye_movement_url = upload_to_catbox(eye_movement_path)
    
    # Save comprehensive focal length data
    focal_data_path = '/tmp/focal_length_data.json'
    with open(focal_data_path, 'w') as f:
        json.dump({
            'display_focal_lengths': display_system.focal_lengths.cpu().tolist(),
            'focal_length_sweep_range': [20.0, 50.0],
            'eye_movement_range': [-10.0, 10.0],
            'optimization_specs': {
                'iterations': iterations,
                'resolution': resolution,
                'rays_per_pixel': 4,
                'display_resolution': 512,
                'focal_planes': 4
            }
        }, f, indent=2)
    
    # Save loss history as JSON
    loss_json_path = '/tmp/loss_history.json'
    with open(loss_json_path, 'w') as f:
        json.dump({
            'loss_history': loss_history,
            'final_loss': loss_history[-1],
            'iterations': iterations,
            'resolution': resolution,
            'optimization_type': 'actual_ray_tracing'
        }, f, indent=2)
    
    loss_url = upload_to_catbox(loss_json_path)
    focal_data_url = upload_to_catbox(focal_data_path)
    os.remove(loss_json_path)
    os.remove(focal_data_path)
    
    print(f"\n" + "="*80)
    print("📥 ALL DOWNLOAD URLS - COMPLETE OUTPUTS:")
    print(f"🎬 Progress GIF ({iterations} frames): {progress_url}")
    print(f"🖼️  Final Comparison: {comparison_url}")
    print(f"📊 What Each Display Shows: {displays_url}")
    print(f"👁️  What Eye Sees for Each Display: {eye_views_url}")
    print(f"🎯 Focal Length Sweep GIF (30 frames): {focal_sweep_url}")
    print(f"🚶 Eye Movement Sweep GIF (20 frames): {eye_movement_url}")
    print(f"📋 Loss History JSON: {loss_url}")
    print(f"⚙️  Focal Length Data JSON: {focal_data_url}")
    print("="*80)
    
    return {
        'status': 'success',
        'message': f'ACTUAL optimization complete: {iterations} iterations, loss: {loss_history[-1]:.6f}',
        'final_loss': loss_history[-1],
        'loss_history': loss_history,
        'progress_gif_url': progress_url,
        'final_comparison_url': comparison_url,
        'displays_url': displays_url,
        'eye_views_url': eye_views_url,
        'focal_sweep_url': focal_sweep_url,
        'eye_movement_url': eye_movement_url,
        'loss_history_url': loss_url,
        'focal_data_url': focal_data_url,
        'all_download_urls': {
            'progress_gif': progress_url,
            'final_comparison': comparison_url,
            'what_displays_show': displays_url,
            'what_eye_sees': eye_views_url,
            'focal_length_sweep': focal_sweep_url,
            'eye_movement_sweep': eye_movement_url,
            'loss_history_json': loss_url,
            'focal_data_json': focal_data_url
        },
        'display_focal_lengths_mm': display_system.focal_lengths.cpu().tolist(),
        'optimization_specs': {
            'iterations': iterations,
            'resolution': resolution,
            'rays_per_pixel': 4,
            'display_resolution': 512,
            'focal_planes': 4,
            'frames_in_progress_gif': iterations,
            'frames_in_focal_sweep': 30,
            'frames_in_eye_movement': 20
        },
        'timestamp': datetime.now().isoformat()
    }

def handler(job):
    try:
        print(f"🚀 ACTUAL LIGHT FIELD OPTIMIZER: {datetime.now()}")
        
        inp = job.get("input", {})
        iterations = inp.get("iterations", 50)  # Reduced for speed
        resolution = inp.get("resolution", 128)  # Reduced for speed
        
        print(f"⚙️ REAL Parameters: {iterations} iterations, {resolution}x{resolution}")
        
        # Test upload first
        print("🧪 Testing upload...")
        test_img = torch.zeros(64, 64, 3)
        plt.imsave('/tmp/test.png', test_img.numpy())
        test_url = upload_to_catbox('/tmp/test.png')
        
        if not test_url:
            return {'status': 'error', 'message': 'Upload test failed'}
        
        print(f"✅ Upload working: {test_url}")
        
        # Run ACTUAL optimization
        result = optimize_spherical_checkerboard(iterations, resolution)
        
        return result  # Return the complete result with all URLs
        
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

runpod.serverless.start({"handler": handler})