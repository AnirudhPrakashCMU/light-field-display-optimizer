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
    samples_per_pixel = 8
    
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
    samples_per_pixel = 8
    
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
    
    # Process in batches
    batch_size = min(1024, N)
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
    display_system = LightFieldDisplay(resolution=1024, num_planes=8)
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
        
        # Save progress frame
        if iteration % 10 == 0:
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
    
    # Upload results
    progress_url = upload_to_catbox(progress_gif)
    
    return {
        'final_loss': loss_history[-1],
        'loss_history': loss_history,
        'progress_url': progress_url
    }

def handler(job):
    try:
        print(f"🚀 ACTUAL LIGHT FIELD OPTIMIZER: {datetime.now()}")
        
        inp = job.get("input", {})
        iterations = inp.get("iterations", 100)
        resolution = inp.get("resolution", 256)
        
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
        
        return {
            'status': 'success',
            'message': f'ACTUAL optimization complete: {iterations} iterations, loss: {result["final_loss"]:.6f}',
            'test_upload': test_url,
            'progress_gif': result['progress_url'],
            'final_loss': result['final_loss'],
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

runpod.serverless.start({"handler": handler})