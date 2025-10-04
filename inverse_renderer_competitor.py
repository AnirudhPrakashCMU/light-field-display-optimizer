#!/usr/bin/env python3
"""
Sphere Synthesis with Focus Tunable Lens - Python Port of MATLAB Script
Inverse Rendering Competitor System

This is a direct Python translation of sphere_synthesis_focus_tunable_standalone.m
Goal: Match MATLAB outputs exactly

Ray tracing flow:
1. DISPLAY GENERATION: Display pixel → MLA → Sphere (inverse ray tracing)
2. VIEWING: Camera → Tunable Lens → MLA → Display (forward ray tracing)

Multi-ray sampling: 8 rays per pixel (matches optimizer) for fair comparison
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import shutil
import math

print("=== SPHERE SYNTHESIS WITH FOCUS TUNABLE LENS (PYTHON) ===")
print("Inverse Rendering Competitor - Direct MATLAB Port")

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Clean and create output directory
output_dir = 'outputs_ft_python'
debug_dir = 'outputs_ft_python/debugging_outputs'
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(debug_dir, exist_ok=True)

class MLAConfig:
    """Microlens Array Configuration - MATCHED TO OPTIMIZER FOR FAIR COMPARISON"""
    def __init__(self):
        # Match optimizer configuration
        self.z0 = 80.0  # MLA position (mm) - matches optimizer microlens_distance
        self.f0 = 1.0    # Microlens focal length (mm) - matches optimizer
        self.disp_z0 = 82.0  # Display position (mm) - matches optimizer display_distance
        self.pitch = 0.4  # Microlens pitch (mm) - matches optimizer microlens_pitch

        # Calculate number of lenses to cover 20mm display
        display_size = 20.0  # Match optimizer display_size
        self.nlens = int(display_size / self.pitch)
        self.width = self.nlens * self.pitch

        # Microlens centers
        self.lens_x = torch.linspace(-(self.nlens-1)/2, (self.nlens-1)/2, self.nlens, device=device) * self.pitch
        self.lens_y = torch.linspace(-(self.nlens-1)/2, (self.nlens-1)/2, self.nlens, device=device) * self.pitch

        # Display grid - MATCHED TO OPTIMIZER (1024x1024)
        self.res = (1024, 1024)
        self.disp_x = torch.linspace(-display_size/2, display_size/2, self.res[1], device=device)
        self.disp_y = torch.linspace(-display_size/2, display_size/2, self.res[0], device=device)
        self.disp_Y, self.disp_X = torch.meshgrid(self.disp_y, self.disp_x, indexing='ij')

        self.disp_img = None

        print(f"MLA Config: z={self.z0}mm, f={self.f0}mm, pitch={self.pitch:.4f}mm, {self.nlens}x{self.nlens} lenses")

class TunableLensConfig:
    """Focus Tunable Lens Configuration - MATCHED TO OPTIMIZER FOR FAIR COMPARISON"""
    def __init__(self):
        self.distance_from_camera = 50.0  # 50mm from camera - matches optimizer tunable_lens_distance
        self.focal_lengths = [30.0, 50.0, 100.0]  # Test focal lengths

        print(f"Tunable Lens: distance={self.distance_from_camera}mm, focal lengths={self.focal_lengths}")

class SphereConfig:
    """Sphere Configuration"""
    def __init__(self, mla, input_img):
        self.img = input_img
        self.center = torch.tensor([0.0, 0.0, mla.z0], device=device)
        self.radius = mla.width / 2.0

        # Spherical coordinate ranges (MATLAB-compatible)
        img_h, img_w = input_img.shape
        self.phiRange = torch.linspace(-math.pi + math.pi/180, -math.pi/180, img_h, device=device)
        self.thetaRange = torch.linspace(math.pi/180, math.pi - math.pi/180, img_w, device=device)

        print(f"Sphere: center={self.center.cpu().numpy()}, radius={self.radius:.2f}mm")

def generate_test_pattern(pattern_type, size=1000, square_size=50):
    """Generate test patterns matching MATLAB"""
    print(f"  Generating {pattern_type} pattern ({size}x{size}, square_size={square_size})...")

    if pattern_type == 'checkerboard':
        img = torch.zeros(size, size, device=device)
        for i in range(0, size, square_size):
            for j in range(0, size, square_size):
                if ((i // square_size) + (j // square_size)) % 2 == 0:
                    i_end = min(i + square_size, size)
                    j_end = min(j + square_size, size)
                    img[i:i_end, j:j_end] = 1.0

    elif pattern_type == 'stripes_horizontal':
        img = torch.zeros(size, size, device=device)
        stripe_width = 50
        for i in range(0, size, stripe_width * 2):
            i_end = min(i + stripe_width, size)
            img[i:i_end, :] = 1.0

    elif pattern_type == 'stripes_vertical':
        img = torch.zeros(size, size, device=device)
        stripe_width = 50
        for j in range(0, size, stripe_width * 2):
            j_end = min(j + stripe_width, size)
            img[:, j:j_end] = 1.0

    elif pattern_type == 'circles':
        img = torch.zeros(size, size, device=device)
        Y, X = torch.meshgrid(torch.arange(size, device=device), torch.arange(size, device=device), indexing='ij')
        for cx in range(50, size, 100):
            for cy in range(50, size, 100):
                mask = torch.sqrt((X - cx)**2 + (Y - cy)**2) < 30
                img[mask] = 1.0

    elif pattern_type == 'gradient':
        img = torch.linspace(0, 1, size, device=device).unsqueeze(1).repeat(1, size)

    else:
        img = torch.zeros(size, size, device=device)

    return img

def compute_ray_sphere_intersection(ray, center, radius):
    """
    Ray-sphere intersection (MATLAB-compatible)
    ray: [6, N] tensor [x, y, z, dx, dy, dz]
    Returns: (dist1, dist2, flag)
    """
    x = ray[:3, :]
    d = ray[3:, :]

    b = torch.sum(d * (x - center.unsqueeze(1)), dim=0)
    c = torch.sum((x - center.unsqueeze(1))**2, dim=0) - radius**2

    discr = b**2 - c
    flag = discr <= 0

    sqrt_discr = torch.sqrt(torch.abs(discr))
    dist1 = -b - sqrt_discr
    dist2 = -b + sqrt_discr

    dist1[flag] = 0
    dist2[flag] = 0

    return dist1, dist2, flag

def compute_ray_plane_intersection(ray, n, c):
    """
    Ray-plane intersection (MATLAB-compatible)
    ray: [6, N] tensor
    n: normal vector [3]
    c: plane offset (scalar)
    """
    x = ray[:3, :]
    d = ray[3:, :]

    rhs = c - torch.sum(n.unsqueeze(1) * x, dim=0)
    lhs = torch.sum(n.unsqueeze(1) * d, dim=0)

    flag = torch.abs(lhs) < 1e-10
    dist = rhs / (lhs + 1e-10)
    dist[flag] = 0

    return dist, flag

def propagate_distance(ray0, dist):
    """Propagate rays by distance (MATLAB-compatible)"""
    ray1 = ray0.clone()
    ray1[:3, :] = ray0[:3, :] + ray0[3:, :] * dist.unsqueeze(0)
    return ray1

def convert_3d_direction_to_euler(X, Y, Z):
    """Convert 3D direction to spherical angles (MATLAB-compatible)"""
    # MATLAB: phi = angle(X + 1j*Z) = atan2(Z, X)
    # MATLAB: theta = angle(Y + 1j*sqrt(X^2+Z^2)) = atan2(sqrt(X^2+Z^2), Y)
    phi = torch.atan2(Z, X)
    theta = torch.atan2(torch.sqrt(X**2 + Z**2), Y)  # Fixed: arguments were swapped
    return theta, phi

def convert_euler_to_3d_direction(theta, phi, rad=1.0):
    """Convert spherical angles to 3D direction (MATLAB-compatible)"""
    X = rad * torch.sin(theta) * torch.cos(phi)
    Y = rad * torch.cos(theta)
    Z = rad * torch.sin(theta) * torch.sin(phi)
    return X, Y, Z

def generate_display_pattern(mla, sphere):
    """
    Generate display pattern using inverse ray tracing
    Display pixel → MLA → Sphere
    """
    print("  Generating display pattern (inverse ray tracing)...")

    # Find nearest microlens for each display pixel
    disp_X_flat = mla.disp_X.flatten()
    disp_Y_flat = mla.disp_Y.flatten()

    # Find closest microlens indices (vectorized)
    x_diffs = torch.abs(mla.lens_x.unsqueeze(0) - disp_X_flat.unsqueeze(1))
    y_diffs = torch.abs(mla.lens_y.unsqueeze(0) - disp_Y_flat.unsqueeze(1))

    x_indx = torch.argmin(x_diffs, dim=1)
    y_indx = torch.argmin(y_diffs, dim=1)

    # Create rays from display pixels to microlens centers
    rays_xyz = torch.stack([
        mla.lens_x[x_indx],
        mla.lens_y[y_indx],
        torch.full_like(disp_X_flat, mla.z0)
    ], dim=0)

    # Ray directions
    rays_dir = rays_xyz - torch.stack([
        disp_X_flat,
        disp_Y_flat,
        torch.full_like(disp_X_flat, mla.disp_z0)
    ], dim=0)

    # Normalize
    rays_dir = rays_dir / torch.norm(rays_dir, dim=0, keepdim=True)

    # Combine to ray format [6, N]
    rays = torch.cat([rays_xyz, rays_dir], dim=0)

    # Ray-sphere intersection
    dist1, dist2, flag = compute_ray_sphere_intersection(rays, sphere.center, sphere.radius)

    # Only process valid intersections
    valid_idx = (~flag) & (dist2 > 0)

    displayImg = torch.zeros(disp_X_flat.shape[0], device=device)

    if valid_idx.any():
        # Propagate rays to intersection
        rays_intersect = propagate_distance(rays[:, valid_idx], dist2[valid_idx])

        # Convert intersection points to spherical coordinates
        rel_x = rays_intersect[0, :] - sphere.center[0]
        rel_y = rays_intersect[1, :] - sphere.center[1]
        rel_z = rays_intersect[2, :] - sphere.center[2]

        sphereTheta, spherePhi = convert_3d_direction_to_euler(rel_x, rel_y, rel_z)

        # Interpolate texture from sphere
        # MATLAB: interp2(thetaRange, phiRange, img, theta, phi)
        # This maps theta values to columns (x-axis) and phi values to rows (y-axis)

        # Find where each theta falls in thetaRange (x-axis)
        # Find where each phi falls in phiRange (y-axis)
        img_h, img_w = sphere.img.shape

        # Map theta to x-coordinate (column index)
        # thetaRange spans columns (width)
        theta_norm = (sphereTheta - sphere.thetaRange[0]) / (sphere.thetaRange[-1] - sphere.thetaRange[0])
        x_coord = theta_norm * (img_w - 1)

        # Map phi to y-coordinate (row index)
        # phiRange spans rows (height)
        phi_norm = (spherePhi - sphere.phiRange[0]) / (sphere.phiRange[-1] - sphere.phiRange[0])
        y_coord = phi_norm * (img_h - 1)

        # Bilinear interpolation
        x0 = torch.floor(x_coord).long().clamp(0, img_w - 1)
        x1 = (x0 + 1).clamp(0, img_w - 1)
        y0 = torch.floor(y_coord).long().clamp(0, img_h - 1)
        y1 = (y0 + 1).clamp(0, img_h - 1)

        wx = x_coord - x0.float()
        wy = y_coord - y0.float()

        val00 = sphere.img[y0, x0]
        val01 = sphere.img[y1, x0]
        val10 = sphere.img[y0, x1]
        val11 = sphere.img[y1, x1]

        valid_values = (1 - wx) * (1 - wy) * val00 + \
                       (1 - wx) * wy * val01 + \
                       wx * (1 - wy) * val10 + \
                       wx * wy * val11

        displayImg[valid_idx] = valid_values

    displayImg = displayImg.reshape(mla.res[0], mla.res[1])
    mla.disp_img = displayImg

    print(f"  Display pixels used: {100 * torch.sum(displayImg > 0) / displayImg.numel():.1f}%")

    return displayImg

def _trace_rays_through_system(rays, mla, cam_pinhole, cam_res, apply_tunable_lens, tunable_focal_length, tunable_distance):
    """Helper function to trace rays through the optical system"""
    # Apply focus tunable lens if requested
    if apply_tunable_lens:
        tunable_lens_z = cam_pinhole[2] + tunable_distance
        dist_to_lens, flag = compute_ray_plane_intersection(rays, torch.tensor([0., 0., 1.], device=device), tunable_lens_z)
        rays_at_lens = propagate_distance(rays, dist_to_lens)

        # Apply thin lens equation
        ray_x = rays_at_lens[0, :]
        ray_y = rays_at_lens[1, :]
        focal_factor = -1.0 / tunable_focal_length

        rays_at_lens[3, :] = rays_at_lens[3, :] + focal_factor * ray_x
        rays_at_lens[4, :] = rays_at_lens[4, :] + focal_factor * ray_y

        # Renormalize
        norm_factors = torch.norm(rays_at_lens[3:, :], dim=0)
        rays_at_lens[3:, :] = rays_at_lens[3:, :] / norm_factors.unsqueeze(0)

        rays = rays_at_lens

    # Propagate to MLA plane
    dist1, flag = compute_ray_plane_intersection(rays, torch.tensor([0., 0., 1.], device=device), mla.z0)
    rays_mla = propagate_distance(rays, dist1)

    # Find nearest microlens
    x_diffs = torch.abs(mla.lens_x.unsqueeze(0) - rays_mla[0, :].unsqueeze(1))
    y_diffs = torch.abs(mla.lens_y.unsqueeze(0) - rays_mla[1, :].unsqueeze(1))

    x_indx = torch.argmin(x_diffs, dim=1)
    y_indx = torch.argmin(y_diffs, dim=1)

    # Calculate display sampling positions
    cam_disp_x = mla.lens_x[x_indx] + mla.f0 * rays_mla[3, :] / rays_mla[5, :]
    cam_disp_y = mla.lens_y[y_indx] + mla.f0 * rays_mla[4, :] / rays_mla[5, :]

    # Interpolate from display image
    # Normalize to grid coordinates
    x_norm = (cam_disp_x - mla.disp_x[0]) / (mla.disp_x[-1] - mla.disp_x[0])
    y_norm = (cam_disp_y - mla.disp_y[0]) / (mla.disp_y[-1] - mla.disp_y[0])

    x_norm = torch.clamp(x_norm, 0, 1)
    y_norm = torch.clamp(y_norm, 0, 1)

    x_coord = x_norm * (mla.res[1] - 1)
    y_coord = y_norm * (mla.res[0] - 1)

    # Bilinear interpolation
    x0 = torch.floor(x_coord).long().clamp(0, mla.res[1] - 1)
    x1 = (x0 + 1).clamp(0, mla.res[1] - 1)
    y0 = torch.floor(y_coord).long().clamp(0, mla.res[0] - 1)
    y1 = (y0 + 1).clamp(0, mla.res[0] - 1)

    wx = x_coord - x0.float()
    wy = y_coord - y0.float()

    val00 = mla.disp_img[y0, x0]
    val01 = mla.disp_img[y1, x0]
    val10 = mla.disp_img[y0, x1]
    val11 = mla.disp_img[y1, x1]

    cam_img = (1 - wx) * (1 - wy) * val00 + \
              (1 - wx) * wy * val01 + \
              wx * (1 - wy) * val10 + \
              wx * wy * val11

    cam_img = cam_img.reshape(cam_res, cam_res)
    return cam_img

def render_camera_view(mla, cam_pinhole, cam_res=1000, apply_tunable_lens=False, tunable_focal_length=50.0, tunable_distance=25.0, samples_per_pixel=8):
    """
    Render camera view through MLA
    Camera → (Optional: Tunable Lens) → MLA → Display

    Args:
        samples_per_pixel: Number of rays per pixel (matches optimizer's 8-ray sampling)
    """
    cam_pinhole = torch.tensor(cam_pinhole, device=device)

    # Camera setup - determine viewing angles
    corners = torch.tensor([
        [mla.disp_x[-1], 0, mla.z0],
        [mla.disp_x[0], 0, mla.z0],
        [0, mla.disp_y[-1], mla.z0],
        [0, mla.disp_y[0], mla.z0]
    ], device=device).T - cam_pinhole.unsqueeze(1)

    cTheta, cPhi = convert_3d_direction_to_euler(corners[0, :], corners[1, :], corners[2, :])

    thetaList = torch.linspace(cTheta.min(), cTheta.max(), cam_res, device=device)
    phiList = torch.linspace(cPhi.min(), cPhi.max(), cam_res, device=device)

    Theta, Phi = torch.meshgrid(thetaList, phiList, indexing='ij')

    # Multi-ray sampling for each pixel (matches optimizer)
    if samples_per_pixel > 1:
        # Jittered sampling within pixel
        pixel_width_theta = (thetaList[1] - thetaList[0]) if len(thetaList) > 1 else 0.0
        pixel_width_phi = (phiList[1] - phiList[0]) if len(phiList) > 1 else 0.0

        accumulated_image = torch.zeros(cam_res, cam_res, device=device)

        for sample_idx in range(samples_per_pixel):
            # Stratified jittering
            offset_theta = (torch.rand(Theta.shape, device=device) - 0.5) * pixel_width_theta
            offset_phi = (torch.rand(Phi.shape, device=device) - 0.5) * pixel_width_phi

            Theta_jittered = Theta + offset_theta
            Phi_jittered = Phi + offset_phi

            # Generate rays from camera
            rays_xyz = cam_pinhole.unsqueeze(1).expand(3, Theta_jittered.numel())
            rays_d1, rays_d2, rays_d3 = convert_euler_to_3d_direction(Theta_jittered.flatten(), Phi_jittered.flatten(), 1.0)

            rays = torch.stack([
                rays_xyz[0, :], rays_xyz[1, :], rays_xyz[2, :],
                rays_d1, rays_d2, rays_d3
            ], dim=0)

            # Ray tracing through optical system
            cam_img_sample = _trace_rays_through_system(rays, mla, cam_pinhole, cam_res,
                                                       apply_tunable_lens, tunable_focal_length, tunable_distance)
            accumulated_image += cam_img_sample

        return accumulated_image / samples_per_pixel

    # Single ray per pixel (original behavior)
    rays_xyz = cam_pinhole.unsqueeze(1).expand(3, Theta.numel())
    rays_d1, rays_d2, rays_d3 = convert_euler_to_3d_direction(Theta.flatten(), Phi.flatten(), 1.0)

    rays = torch.stack([
        rays_xyz[0, :], rays_xyz[1, :], rays_xyz[2, :],
        rays_d1, rays_d2, rays_d3
    ], dim=0)

    cam_img = _trace_rays_through_system(rays, mla, cam_pinhole, cam_res,
                                         apply_tunable_lens, tunable_focal_length, tunable_distance)

    return cam_img

def process_pattern(pattern_type, square_size=50, save_debug=True):
    """Process a single test pattern through the full pipeline

    Returns:
        cam_img: Camera view at nominal position (x=2mm, f=100mm) for GIF creation
    """
    # Determine pattern name with square count
    if pattern_type == 'checkerboard':
        num_squares = 1000 // square_size  # Calculate number of squares
        pattern_name = f"{pattern_type}_{num_squares}x{num_squares}"
        print(f"\nProcessing {pattern_type} pattern ({num_squares}x{num_squares} squares)...")
    else:
        pattern_name = pattern_type
        print(f"\nProcessing {pattern_type} pattern...")

    # Setup
    mla = MLAConfig()
    tunable = TunableLensConfig()

    # Generate test pattern
    input_img = generate_test_pattern(pattern_type, square_size=square_size)

    # Save input to debug folder
    if save_debug:
        plt.figure(figsize=(8, 6))
        plt.imshow(input_img.cpu().numpy(), cmap='gray')
        plt.title(f'Input Image: {pattern_name.replace("_", " ").title()}')
        plt.axis('off')
        plt.savefig(f'{debug_dir}/{pattern_name}_input.png', dpi=150, bbox_inches='tight')
        plt.close()

    # Setup sphere
    sphere = SphereConfig(mla, input_img)

    # Generate display pattern (INVERSE RAY TRACING)
    display_img = generate_display_pattern(mla, sphere)

    # Save display pattern to debug folder
    if save_debug:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(display_img.cpu().numpy(), cmap='gray')
        axes[0].set_title('Full Display')
        axes[0].axis('off')

        # Center crop
        center_size = min(100, mla.res[0] // 2)
        center_start = (mla.res[0] - center_size) // 2
        center_crop = display_img[center_start:center_start+center_size, center_start:center_start+center_size]
        axes[1].imshow(center_crop.cpu().numpy(), cmap='gray')
        axes[1].set_title('Center Region')
        axes[1].axis('off')

        # Zoom
        zoom_size = min(20, mla.res[0] // 4)
        zoom_start = (mla.res[0] - zoom_size) // 2
        zoom_crop = display_img[zoom_start:zoom_start+zoom_size, zoom_start:zoom_start+zoom_size]
        axes[2].imshow(zoom_crop.cpu().numpy(), cmap='gray')
        axes[2].set_title('Zoomed Detail')
        axes[2].axis('off')

        plt.suptitle(f'Display Pattern: {pattern_name.replace("_", " ").title()}')
        plt.savefig(f'{debug_dir}/{pattern_name}_display.png', dpi=150, bbox_inches='tight')
        plt.close()

    # Render views WITHOUT tunable lens
    print("  Rendering views without tunable lens...")
    camera_positions = [-4, -2, 0, 2, 4]
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))

    for idx, cam_x in enumerate(camera_positions):
        cam_img = render_camera_view(mla, [cam_x, 0, 0], cam_res=200, apply_tunable_lens=False)
        axes[idx // 3, idx % 3].imshow(cam_img.cpu().numpy(), cmap='gray')
        axes[idx // 3, idx % 3].set_title(f'Camera x={cam_x}mm')
        axes[idx // 3, idx % 3].axis('off')

    axes[1, 2].imshow(center_crop.cpu().numpy(), cmap='gray')
    axes[1, 2].set_title('Display Pattern')
    axes[1, 2].axis('off')

    plt.suptitle(f'Multiple Views WITHOUT Focus Tunable: {pattern_name.replace("_", " ").title()}')
    plt.tight_layout()
    plt.savefig(f'{debug_dir}/{pattern_name}_views_no_ft.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Render views WITH tunable lens (save to debug)
    if save_debug:
        for focal_length in tunable.focal_lengths:
            print(f"  Rendering with focal length {focal_length}mm...")
        fig, axes = plt.subplots(2, 3, figsize=(10, 6))

        for idx, cam_x in enumerate(camera_positions):
            cam_img = render_camera_view(mla, [cam_x, 0, 0], cam_res=200,
                                        apply_tunable_lens=True,
                                        tunable_focal_length=focal_length,
                                        tunable_distance=tunable.distance_from_camera)
            axes[idx // 3, idx % 3].imshow(cam_img.cpu().numpy(), cmap='gray')
            axes[idx // 3, idx % 3].set_title(f'Camera x={cam_x}mm')
            axes[idx // 3, idx % 3].axis('off')

        axes[1, 2].imshow(center_crop.cpu().numpy(), cmap='gray')
        axes[1, 2].set_title('Display Pattern')
        axes[1, 2].axis('off')

        plt.suptitle(f'WITH Focus Tunable (f={focal_length:.0f}mm): {pattern_name.replace("_", " ").title()}')
        plt.tight_layout()
        plt.savefig(f'{debug_dir}/{pattern_name}_views_ft_{int(focal_length)}mm.png', dpi=150, bbox_inches='tight')
        plt.close()

    # Generate focal sweep animation (save to debug)
    if save_debug:
        print("  Generating focus tunable animation...")
        focal_sweep = list(range(20, 101, 5)) + list(range(95, 24, -5))  # Sweep focal lengths
        frames = []

        for frame_idx, f in enumerate(focal_sweep):
            cam_img = render_camera_view(mla, [0, 0, 0], cam_res=600,
                                         apply_tunable_lens=True,
                                         tunable_focal_length=float(f),
                                         tunable_distance=tunable.distance_from_camera)

            fig = plt.figure(figsize=(6, 6))
            plt.imshow(cam_img.cpu().numpy(), cmap='gray')
            plt.title(f'{pattern_name.replace("_", " ").title()} - Focal Length: {f}mm', fontsize=14)
            plt.axis('off')

            frame_path = f'{debug_dir}/{pattern_name}_focus_frame_{frame_idx:03d}.png'
            plt.savefig(frame_path, dpi=100, bbox_inches='tight')
            frames.append(frame_path)
            plt.close()

        # Create GIF
        gif_filename = f'{debug_dir}/{pattern_name}_focus_animation.gif'
        images = [Image.open(frame) for frame in frames]
        images[0].save(gif_filename, save_all=True, append_images=images[1:],
                       duration=100, loop=0, optimize=True)

        # Clean up frames
        for frame in frames:
            os.remove(frame)

        print(f"  Created {gif_filename}")

    # Generate camera movement animation (MATLAB lines 333-398)
    print("  Generating standard animation with optimal focus...")
    camera_x_positions = list(np.linspace(-5, 5, 30))
    f = 50.0  # Optimal focal length
    frames = []

    for frame_idx, cam_x in enumerate(camera_x_positions):
        cam_img = render_camera_view(mla, [cam_x, 0, 0], cam_res=600,
                                     apply_tunable_lens=True,
                                     tunable_focal_length=f,
                                     tunable_distance=tunable.distance_from_camera)

        fig = plt.figure(figsize=(6, 6))
        plt.imshow(cam_img.cpu().numpy(), cmap='gray')
        plt.title(f'{pattern_name.replace("_", " ").title()} - Camera x={cam_x:.1f}mm', fontsize=14)
        plt.axis('off')

        frame_path = f'{debug_dir}/{pattern_name}_animation_frame_{frame_idx:03d}.png'
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        frames.append(frame_path)
        plt.close()

    # Create GIF
    gif_filename = f'{debug_dir}/{pattern_name}_animation.gif'
    images = [Image.open(frame) for frame in frames]
    images[0].save(gif_filename, save_all=True, append_images=images[1:],
                   duration=100, loop=0, optimize=True)

    # Clean up frames
    for frame in frames:
        os.remove(frame)

    print(f"  Created {gif_filename}")

    # Render nominal view for main GIF (x=2mm, f=100mm)
    nominal_view = render_camera_view(mla, [2, 0, 0], cam_res=512,
                                      apply_tunable_lens=True,
                                      tunable_focal_length=100.0,
                                      tunable_distance=tunable.distance_from_camera)

    return nominal_view, num_squares

def main():
    """Main execution"""
    print("\n=== GENERATING CHECKERBOARD DENSITY SWEEP ===")
    print("Nominal viewpoint: x=2mm, f=100mm")
    print("Sweeping checkerboard from 26x26 to 60x60 squares\n")

    # Checkerboard configurations: 26x26 to 60x60 (increment by 2)
    frames = []
    frame_info = []

    for num_squares in range(25, 61, 5):  # 25, 30, 35, ..., 60
        square_size = 1000 // num_squares

        # Process pattern and get nominal view
        nominal_view, actual_squares = process_pattern('checkerboard', square_size=square_size, save_debug=True)

        # Save frame for main GIF
        fig = plt.figure(figsize=(8, 8))
        plt.imshow(nominal_view.cpu().numpy(), cmap='gray')
        plt.title(f'Checkerboard {actual_squares}x{actual_squares} - Nominal View (x=2mm, f=100mm)', fontsize=16)
        plt.axis('off')

        frame_path = f'{output_dir}/main_frame_{len(frames):03d}.png'
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        frames.append(frame_path)
        frame_info.append(actual_squares)
        plt.close()

    # Create main GIF (repeat each frame 10 times for slower playback)
    print("\n=== CREATING MAIN GIF ===")

    # Create comparatives folder
    comparatives_dir = 'comparatives'
    os.makedirs(comparatives_dir, exist_ok=True)

    gif_filename = f'{comparatives_dir}/competitor_density_sweep.gif'
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
    print(f"\n=== COMPLETE ===")
    print(f"Main output: {gif_filename}")
    print(f"  • Sweeps from {frame_info[0]}x{frame_info[0]} to {frame_info[-1]}x{frame_info[-1]} checkerboard")
    print(f"  • Nominal viewpoint: camera x=2mm, focal length f=100mm")
    print(f"\nDebug outputs: {debug_dir}/")
    print(f"  • *_input.png - Input patterns")
    print(f"  • *_display.png - Generated displays (inverse ray tracing)")
    print(f"  • *_views_no_ft.png - Views without tunable lens")
    print(f"  • *_views_ft_*mm.png - Views with tunable lens (30mm, 50mm, 100mm)")
    print(f"  • *_focus_animation.gif - Focal length sweep animations")
    print(f"  • *_animation.gif - Camera movement animations")

if __name__ == "__main__":
    main()
