#!/usr/bin/env python3
"""
Spherical Checkerboard Ray Tracer - Complete Implementation
Forward ray tracing from tilted eye retina to physical spherical checkerboard
Pure ray tracing - blur emerges naturally from sub-aperture sampling
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import shutil
import math

print("=== SPHERICAL CHECKERBOARD RAY TRACER ===")
print("Forward ray tracing: Tilted Retina → Eye Lens → Physical Scene")
print("Pure ray tracing (no computational blur)")

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Clean results
if os.path.exists('results'):
    shutil.rmtree('results')
os.makedirs('results', exist_ok=True)

class EyeParams:
    """Eye parameters"""
    pupil_diameter = 4.0  # mm
    retina_distance = 24.0  # mm
    retina_size = 8.0  # mm
    samples_per_pixel = 8  # 8 rays per pixel (matches optimizer and competitor)
    focal_range = (20.0, 50.0)  # mm

class SphericalCheckerboard:
    """Physical spherical checkerboard scene"""
    
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius
        print(f"Spherical Checkerboard: center={center.cpu().numpy()}, radius={radius}mm")
        
    def get_color(self, point_3d, square_size=50):
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

        # Map to flat checkerboard pattern (1000x1000, variable square size)
        theta_norm = (theta + math.pi/2) / math.pi
        phi_norm = (phi + math.pi) / (2*math.pi)

        i_coord = theta_norm * 999
        j_coord = phi_norm * 999

        i_square = torch.floor(i_coord / square_size).long()
        j_square = torch.floor(j_coord / square_size).long()

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

def render_eye_view(eye_position, eye_focal_length, scene, params, square_size=50, resolution=600):
    """
    Render eye view with tilted retina pointing at sphere center
    PURE RAY TRACING - blur from sub-aperture ray averaging only
    """
    
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
    retina_size = params.retina_size
    u_coords = torch.linspace(-retina_size/2, retina_size/2, resolution, device=device)
    v_coords = torch.linspace(-retina_size/2, retina_size/2, resolution, device=device)
    u_grid, v_grid = torch.meshgrid(u_coords, v_coords, indexing='ij')
    
    # Retina center positioned behind eye, along forward direction
    retina_center = eye_position - params.retina_distance * forward_dir
    
    # Retina points in tilted plane
    retina_points = (retina_center.unsqueeze(0).unsqueeze(0) + 
                    u_grid.unsqueeze(-1) * right_dir.unsqueeze(0).unsqueeze(0) +
                    v_grid.unsqueeze(-1) * up_dir.unsqueeze(0).unsqueeze(0))
    
    retina_points_flat = retina_points.reshape(-1, 3)
    N = retina_points_flat.shape[0]
    M = params.samples_per_pixel
    
    # Generate pupil samples
    pupil_radius = params.pupil_diameter / 2
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
        valid_colors = scene.get_color(valid_intersections, square_size=square_size)
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
                # PURE AVERAGING - no computational blur, just ray sampling
                pixel_colors = colors[pixel_idx, valid_samples, :]
                final_colors[pixel_idx, :] = torch.mean(pixel_colors, dim=0)
    
    return final_colors.reshape(resolution, resolution, 3)

def create_standard_eye_view():
    """Create single clean eye view - just what the eye sees"""
    print("Creating standard eye view...")
    
    scene = SphericalCheckerboard(
        center=torch.tensor([0.0, 0.0, 200.0], device=device),
        radius=50.0
    )
    
    eye_params = EyeParams()
    eye_position = torch.tensor([0.0, 0.0, 0.0], device=device)
    eye_focal_length = 30.0  # Standard focal length
    
    eye_view = render_eye_view(eye_position, eye_focal_length, scene, eye_params, resolution=512)
    
    # Save pure eye view
    plt.figure(figsize=(8, 8))
    plt.imshow(eye_view.cpu().numpy())
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('results/standard_eye_view.png', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    print("Standard eye view saved")

def create_focal_length_comparison():
    """Create side-by-side comparison of 4 focal lengths"""
    print("Creating focal length comparison...")
    
    scene = SphericalCheckerboard(
        center=torch.tensor([0.0, 0.0, 200.0], device=device),
        radius=50.0
    )
    
    eye_params = EyeParams()
    eye_position = torch.tensor([0.0, 0.0, 0.0], device=device)
    focal_lengths = [25.0, 30.0, 35.0, 40.0]
    
    plt.figure(figsize=(16, 4))
    
    for i, focal_length in enumerate(focal_lengths):
        eye_view = render_eye_view(eye_position, focal_length, scene, eye_params, resolution=256)
        
        # Calculate focus status
        focused_distance = (focal_length * eye_params.retina_distance) / (focal_length - eye_params.retina_distance)
        defocus_distance = abs(200.0 - focused_distance)
        
        plt.subplot(1, 4, i + 1)
        plt.imshow(eye_view.cpu().numpy())
        
        if defocus_distance < 10:
            status = "SHARP"
            color = 'green'
        elif defocus_distance < 25:
            status = "BLUR"
            color = 'orange'
        else:
            status = "HEAVY BLUR"
            color = 'red'
        
        plt.title(f'FL: {focal_length:.0f}mm\\n{status}', fontsize=14, color=color, fontweight='bold')
        plt.axis('off')
    
    plt.suptitle('Focal Length Comparison - Pure Ray Tracing\\nTilted Eye → Spherical Checkerboard at 200mm', fontsize=16)
    plt.tight_layout()
    plt.savefig('results/focal_length_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Focal length comparison saved")

def create_focal_length_sweep_gif():
    """Create focal length sweep GIF"""
    print("Creating focal length sweep GIF...")
    
    scene = SphericalCheckerboard(
        center=torch.tensor([0.0, 0.0, 200.0], device=device),
        radius=50.0
    )
    
    eye_params = EyeParams()
    eye_position = torch.tensor([0.0, 0.0, 0.0], device=device)
    
    # Focal length sweep
    focal_lengths = torch.linspace(20.0, 50.0, 50, device=device)
    
    frames = []
    for i, focal_length in enumerate(focal_lengths):
        eye_view = render_eye_view(eye_position, focal_length.item(), scene, eye_params, resolution=256)
        
        # Calculate focus parameters
        focused_distance = (focal_length.item() * eye_params.retina_distance) / (focal_length.item() - eye_params.retina_distance)
        defocus_distance = abs(200.0 - focused_distance)
        
        plt.figure(figsize=(10, 8))
        
        # Main eye view (large)
        plt.subplot(2, 1, 1)
        plt.imshow(eye_view.cpu().numpy())
        plt.title(f'Tilted Eye View - FL: {focal_length:.1f}mm\\nPure Ray Tracing (No Computational Blur)', fontsize=16)
        plt.axis('off')
        
        # Focus status
        plt.subplot(2, 1, 2)
        plt.axis('off')
        
        if defocus_distance < 10:
            status = "SHARP FOCUS"
            color = 'green'
        elif defocus_distance < 25:
            status = "MODERATE BLUR"
            color = 'orange'
        else:
            status = "HEAVY BLUR"
            color = 'red'
        
        plt.text(0.5, 0.8, f'Focal Length: {focal_length:.1f}mm', ha='center', fontsize=18, fontweight='bold')
        plt.text(0.5, 0.6, f'Focus Distance: {focused_distance:.0f}mm', ha='center', fontsize=14)
        plt.text(0.5, 0.4, f'Sphere Distance: 200mm', ha='center', fontsize=14)
        plt.text(0.5, 0.2, status, ha='center', fontsize=16, color=color, fontweight='bold')
        
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        
        plt.suptitle(f'Focal Length Sweep - Pure Ray Tracing (Frame {i+1}/50)', fontsize=18)
        plt.tight_layout()
        
        frame_path = f'results/focal_sweep_frame_{i:03d}.png'
        plt.savefig(frame_path, dpi=120, bbox_inches='tight')
        frames.append(frame_path)
        plt.close()
        
        if i % 10 == 0:
            print(f"  Generated focal sweep frame {i+1}/50 (FL: {focal_length:.1f}mm)")
    
    # Create GIF with timing based on focus quality
    images = []
    durations = []
    
    for i, focal_length in enumerate(focal_lengths):
        focused_distance = (focal_length.item() * eye_params.retina_distance) / (focal_length.item() - eye_params.retina_distance)
        defocus_distance = abs(200.0 - focused_distance)
        
        images.append(Image.open(frames[i]))
        
        # Slow down when in focus
        if defocus_distance < 10:
            durations.append(800)  # Long pause when sharp
        elif defocus_distance < 25:
            durations.append(300)  # Medium pause
        else:
            durations.append(150)  # Normal speed when blurred
    
    images[0].save('results/focal_length_sweep.gif',
                  save_all=True, append_images=images[1:],
                  duration=durations, loop=0, optimize=True)
    
    # Clean up frames
    for frame in frames:
        os.remove(frame)
    
    print("Focal length sweep GIF created!")

def create_eye_movement_gif():
    """Create eye movement GIF with tilted retina"""
    print("Creating eye movement GIF...")
    
    scene = SphericalCheckerboard(
        center=torch.tensor([0.0, 0.0, 200.0], device=device),
        radius=50.0
    )
    
    eye_params = EyeParams()
    eye_focal_length = 30.0  # Fixed focus
    
    # Eye positions
    x_positions = torch.linspace(-20, 20, 30, device=device)
    
    frames = []
    for i, x_pos in enumerate(x_positions):
        eye_position = torch.tensor([x_pos.item(), 0.0, 0.0], device=device)
        
        eye_view = render_eye_view(eye_position, eye_focal_length, scene, eye_params, resolution=256)
        
        plt.figure(figsize=(8, 8))
        plt.imshow(eye_view.cpu().numpy())
        plt.title(f'Tilted Eye View - X: {x_pos:.1f}mm\\nRetina Points at Sphere Center', fontsize=14)
        plt.axis('off')
        
        frame_path = f'results/eye_movement_frame_{i:03d}.png'
        plt.savefig(frame_path, dpi=120, bbox_inches='tight')
        frames.append(frame_path)
        plt.close()
        
        if i % 6 == 0:
            print(f"  Generated eye movement frame {i+1}/30")
    
    images = [Image.open(frame) for frame in frames]
    images[0].save('results/eye_movement_sweep.gif',
                  save_all=True, append_images=images[1:],
                  duration=200, loop=0, optimize=True)
    
    # Clean up frames
    for frame in frames:
        os.remove(frame)
    
    print("Eye movement GIF created!")

def create_checkerboard_density_sweep():
    """Create checkerboard density sweep GIF matching competitor"""
    print("\n=== GENERATING CHECKERBOARD DENSITY SWEEP (GROUND TRUTH) ===")
    print("Nominal viewpoint: x=2mm, f=30mm (focused at 200mm)")
    print("Sweeping checkerboard from 26x26 to 60x60 squares\n")

    scene = SphericalCheckerboard(
        center=torch.tensor([0.0, 0.0, 200.0], device=device),
        radius=50.0
    )

    eye_params = EyeParams()
    eye_position = torch.tensor([2.0, 0.0, 0.0], device=device)  # x=2mm to match competitor
    eye_focal_length = 30.0  # Focused at 200mm

    frames = []
    frame_info = []

    for num_squares in range(26, 62, 2):  # 26, 28, 30, ..., 60
        square_size = 1000 // num_squares
        actual_squares = 1000 // square_size

        print(f"Processing {actual_squares}x{actual_squares} checkerboard...")

        eye_view = render_eye_view(eye_position, eye_focal_length, scene, eye_params,
                                   square_size=square_size, resolution=512)

        # Save frame
        fig = plt.figure(figsize=(8, 8))
        plt.imshow(eye_view.cpu().numpy())
        plt.title(f'Ground Truth: Checkerboard {actual_squares}x{actual_squares}\nDirect Eye View (x=2mm, f=30mm)', fontsize=16)
        plt.axis('off')

        frame_path = f'results/gt_frame_{len(frames):03d}.png'
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        frames.append(frame_path)
        frame_info.append(actual_squares)
        plt.close()

    # Create GIF (repeat each frame 10 times for slower playback)
    print("\n=== CREATING GROUND TRUTH GIF ===")

    # Create comparatives folder
    comparatives_dir = 'comparatives'
    os.makedirs(comparatives_dir, exist_ok=True)

    gif_filename = f'{comparatives_dir}/ground_truth_density_sweep.gif'
    images = [Image.open(frame) for frame in frames]

    # Repeat each frame 10 times
    repeated_images = []
    for img in images:
        for _ in range(10):
            repeated_images.append(img.copy())

    repeated_images[0].save(gif_filename, save_all=True, append_images=repeated_images[1:],
                           duration=200, loop=0, optimize=True)

    # Clean up frames
    for frame in frames:
        os.remove(frame)

    print(f"✓ Created {gif_filename}")
    print(f"  • Sweeps from {frame_info[0]}x{frame_info[0]} to {frame_info[-1]}x{frame_info[-1]} checkerboard")
    print(f"  • Ground truth: Direct eye view (no display, no MLA)")
    print(f"  • Eye position: x=2mm, focal length f=30mm")

def main():
    """Main function"""
    print("\nConfiguration:")
    print("  • Tilted retina: Always points at sphere center")
    print("  • Pure ray tracing: Single ray per pixel (no blur)")
    print("  • Perfect pinhole camera model")
    print("  • Spherical checkerboard: MATLAB-compatible (1000x1000, variable squares)")
    print("  • Rendering resolution: 512x512 pixels (matches optimizer)")

    print("\nGenerating ground truth checkerboard density sweep...")

    # Create density sweep GIF
    create_checkerboard_density_sweep()

    print("\n=== COMPLETE ===")
    print("Generated files:")
    print("  • comparatives/ground_truth_density_sweep.gif - Checkerboard density sweep (26x26 to 60x60)")
    print("\nKey features:")
    print("  ✓ Ground truth: Direct eye-to-scene ray tracing")
    print("  ✓ No display, no MLA - just physical spherical checkerboard")
    print("  ✓ Matches competitor nominal viewpoint (x=2mm)")
    print("  ✓ High resolution (600x600 pixels)")
    print("  ✓ Same checkerboard densities as competitor")

if __name__ == "__main__":
    main()