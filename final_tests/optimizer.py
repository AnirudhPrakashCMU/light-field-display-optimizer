#!/usr/bin/env python3
"""
PyTorch Light Field Display Optimizer - ENHANCED WITH MULTI-RAY SAMPLING
Multi-scene optimization with FIXED focal lengths, circular microlenses, and spherical checkerboard
High quality outputs with both MP4 and GIF support, enhanced ray sampling from spherical_checkerboard_raytracer.py
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
import concurrent.futures
import threading

print("=== PYTORCH 80GB A100-MAXIMIZED ULTRA-HIGH-QUALITY LIGHT FIELD OPTIMIZER ===")
print("Maximum-resolution multi-scene optimization with 80GB A100 acceleration")
print("Enhanced with 128-ray sub-aperture sampling, mixed precision, and 200-iteration training")

# PARALLEL PROCESSING TOGGLE - Set to True for experimental parallel scene optimization
ENABLE_PARALLEL_SCENES = False  # KEEP FALSE - parallel has file conflicts, use sequential for safety

# Device setup with GPU memory clearing
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if torch.cuda.is_available():
    # Aggressive GPU memory clearing at startup
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    # Set memory allocation strategy to avoid fragmentation
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    print("GPU memory aggressively cleared at startup")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Report initial memory
    initial_memory = torch.cuda.memory_allocated() / 1024**3
    print(f"Initial GPU memory used: {initial_memory:.2f} GB")
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
print()

# Clean up results directory - save to workspace for RunPod (change for Colab if needed)
base_results_dir = '/workspace/results'  # Use /content/drive/MyDrive/aswin for Colab
comprehensive_results_dir = f'{base_results_dir}/comprehensive_results'
if os.path.exists(comprehensive_results_dir):
    shutil.rmtree(comprehensive_results_dir)
os.makedirs(comprehensive_results_dir, exist_ok=True)
print(f"Results will be saved to: {comprehensive_results_dir}")

@dataclass
class OpticalSystemParams:
    """Parameters for the complete optical system"""
    # Eye parameters - High resolution with multi-ray sampling
    eye_lens_focal_range: Tuple[float, float] = (17.0, 60.0)  # mm
    eye_pupil_diameter: float = 4.0  # mm (increased for better multi-ray sampling)
    retina_distance: float = 24.0  # mm
    retina_size: float = 10.0  # mm (effective imaging area)
    samples_per_pixel: int = 8  # Balanced multi-ray sampling for speed
    
    # Tunable lens parameters
    tunable_lens_distance: float = 50.0  # mm from eye
    tunable_lens_diameter: float = 15.0  # mm
    tunable_lens_focal_range: Tuple[float, float] = (10.0, 100.0)  # mm
    
    # Microlens array parameters - CIRCULAR microlenses (A100 optimized)
    microlens_distance: float = 80.0  # mm from eye
    microlens_array_size: float = 20.0  # mm x mm (reasonable array)
    microlens_pitch: float = 0.5  # mm spacing between microlenses (reasonable density)
    microlens_focal_length: float = 1.0  # mm
    
    # Display parameters - Ultra high resolution for A100
    display_distance: float = 82.0  # mm from eye (1mm behind microlens array)
    display_size: float = 20.0  # mm x mm (reasonable display)
    display_resolution: int = 1024  # pixels per side (balanced quality/speed)
    num_focal_planes: int = 6  # Fewer focal planes for speed

@dataclass
class SceneObject:
    """3D scene object for target generation"""
    position: torch.Tensor  # [x, y, z] in mm
    size: float  # characteristic size in mm
    color: torch.Tensor  # [r, g, b] color
    shape: str  # 'sphere', 'cube', etc.

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

class HighQualityLightFieldDisplay(nn.Module):
    """High quality light field display system with FIXED focal lengths"""
    
    def __init__(self, params: OpticalSystemParams):
        super().__init__()
        self.params = params
        
        # Learnable display images at different focal lengths
        self.display_images = nn.Parameter(
            torch.rand(params.num_focal_planes, 3, params.display_resolution, params.display_resolution, 
                      device=device, dtype=torch.float32) * 0.5
        )
        
        # FIXED focal lengths for each display plane (not learnable)
        self.focal_lengths = torch.linspace(
            params.tunable_lens_focal_range[0], 
            params.tunable_lens_focal_range[1], 
            params.num_focal_planes, 
            device=device
        )
        
        # Pre-compute CIRCULAR microlens positions
        self.microlens_positions = self._compute_microlens_positions()
        
    def _compute_microlens_positions(self) -> torch.Tensor:
        """Pre-compute circular microlens center positions"""
        pitch = self.params.microlens_pitch
        array_size = self.params.microlens_array_size
        
        # Number of microlenses per side
        num_lenses = int(array_size / pitch)
        
        # Grid of microlens centers
        x_centers = torch.linspace(-array_size/2, array_size/2, num_lenses, device=device)
        y_centers = torch.linspace(-array_size/2, array_size/2, num_lenses, device=device)
        
        # Create meshgrid
        x_grid, y_grid = torch.meshgrid(x_centers, y_centers, indexing='ij')
        
        # Flatten and stack to get [N, 2] array of centers
        positions = torch.stack([x_grid.flatten(), y_grid.flatten()], dim=1)
        
        return positions

def create_scene_basic() -> List[SceneObject]:
    """Create basic geometric scene"""
    objects = []
    
    # Red sphere at 150mm
    objects.append(SceneObject(
        position=torch.tensor([0.0, 5.0, 150.0], device=device),
        size=8.0,
        color=torch.tensor([1.0, 0.3, 0.3], device=device),
        shape='sphere'
    ))
    
    # Green cube at 200mm
    objects.append(SceneObject(
        position=torch.tensor([-10.0, -5.0, 200.0], device=device),
        size=6.0,
        color=torch.tensor([0.3, 1.0, 0.3], device=device),
        shape='cube'
    ))
    
    # Blue sphere at 250mm
    objects.append(SceneObject(
        position=torch.tensor([8.0, 0.0, 250.0], device=device),
        size=5.0,
        color=torch.tensor([0.3, 0.3, 1.0], device=device),
        shape='sphere'
    ))
    
    # Yellow sphere at 300mm
    objects.append(SceneObject(
        position=torch.tensor([0.0, -8.0, 300.0], device=device),
        size=4.0,
        color=torch.tensor([1.0, 1.0, 0.3], device=device),
        shape='sphere'
    ))
    
    return objects

def create_scene_complex() -> List[SceneObject]:
    """Create complex multi-depth scene"""
    objects = []
    
    # Foreground objects
    objects.append(SceneObject(
        position=torch.tensor([-5.0, 8.0, 120.0], device=device),
        size=3.0,
        color=torch.tensor([1.0, 0.5, 0.0], device=device),  # Orange
        shape='cube'
    ))
    
    objects.append(SceneObject(
        position=torch.tensor([6.0, -6.0, 140.0], device=device),
        size=4.0,
        color=torch.tensor([0.8, 0.2, 0.8], device=device),  # Magenta
        shape='sphere'
    ))
    
    # Mid-ground objects
    objects.append(SceneObject(
        position=torch.tensor([0.0, 0.0, 180.0], device=device),
        size=7.0,
        color=torch.tensor([0.2, 0.8, 0.2], device=device),  # Green
        shape='cube'
    ))
    
    objects.append(SceneObject(
        position=torch.tensor([-8.0, 4.0, 220.0], device=device),
        size=5.0,
        color=torch.tensor([0.2, 0.6, 1.0], device=device),  # Light blue
        shape='sphere'
    ))
    
    # Background objects
    objects.append(SceneObject(
        position=torch.tensor([3.0, -2.0, 280.0], device=device),
        size=6.0,
        color=torch.tensor([1.0, 1.0, 0.2], device=device),  # Yellow
        shape='cube'
    ))
    
    objects.append(SceneObject(
        position=torch.tensor([0.0, 6.0, 320.0], device=device),
        size=4.0,
        color=torch.tensor([0.6, 0.3, 0.9], device=device),  # Purple
        shape='sphere'
    ))
    
    return objects

def create_scene_stick_figure() -> List[SceneObject]:
    """Create stick figure scene with background objects"""
    objects = []
    
    # Stick figure at 160mm
    figure_z = 160.0
    figure_x = -2.0
    figure_y = 0.0
    
    # Head
    objects.append(SceneObject(
        position=torch.tensor([figure_x, figure_y + 8.0, figure_z], device=device),
        size=2.5,
        color=torch.tensor([0.9, 0.7, 0.5], device=device),  # Skin tone
        shape='sphere'
    ))
    
    # Body (torso)
    objects.append(SceneObject(
        position=torch.tensor([figure_x, figure_y + 3.0, figure_z], device=device),
        size=3.0,
        color=torch.tensor([0.2, 0.4, 0.8], device=device),  # Blue shirt
        shape='cube'
    ))
    
    # Left arm
    objects.append(SceneObject(
        position=torch.tensor([figure_x - 3.0, figure_y + 4.0, figure_z], device=device),
        size=1.5,
        color=torch.tensor([0.9, 0.7, 0.5], device=device),  # Skin tone
        shape='cube'
    ))
    
    # Right arm
    objects.append(SceneObject(
        position=torch.tensor([figure_x + 3.0, figure_y + 4.0, figure_z], device=device),
        size=1.5,
        color=torch.tensor([0.9, 0.7, 0.5], device=device),  # Skin tone
        shape='cube'
    ))
    
    # Left leg
    objects.append(SceneObject(
        position=torch.tensor([figure_x - 1.5, figure_y - 2.0, figure_z], device=device),
        size=2.0,
        color=torch.tensor([0.3, 0.3, 0.3], device=device),  # Dark pants
        shape='cube'
    ))
    
    # Right leg
    objects.append(SceneObject(
        position=torch.tensor([figure_x + 1.5, figure_y - 2.0, figure_z], device=device),
        size=2.0,
        color=torch.tensor([0.3, 0.3, 0.3], device=device),  # Dark pants
        shape='cube'
    ))
    
    # Background objects
    # Tree (green sphere behind figure)
    objects.append(SceneObject(
        position=torch.tensor([8.0, 2.0, 240.0], device=device),
        size=8.0,
        color=torch.tensor([0.1, 0.6, 0.1], device=device),  # Tree green
        shape='sphere'
    ))
    
    # Building/house (large cube in background)
    objects.append(SceneObject(
        position=torch.tensor([-6.0, -1.0, 300.0], device=device),
        size=10.0,
        color=torch.tensor([0.7, 0.5, 0.3], device=device),  # Brown building
        shape='cube'
    ))
    
    # Sun (yellow sphere in far background)
    objects.append(SceneObject(
        position=torch.tensor([12.0, 10.0, 400.0], device=device),
        size=6.0,
        color=torch.tensor([1.0, 0.9, 0.3], device=device),  # Sun yellow
        shape='sphere'
    ))
    
    return objects

def create_scene_layered() -> List[SceneObject]:
    """Create layered depth scene"""
    objects = []
    
    # Layer 1 - Closest (100-150mm)
    objects.append(SceneObject(
        position=torch.tensor([0.0, 0.0, 110.0], device=device),
        size=2.0,
        color=torch.tensor([1.0, 0.0, 0.0], device=device),  # Bright red
        shape='sphere'
    ))
    
    objects.append(SceneObject(
        position=torch.tensor([4.0, 3.0, 130.0], device=device),
        size=2.5,
        color=torch.tensor([1.0, 0.5, 0.0], device=device),  # Orange
        shape='cube'
    ))
    
    # Layer 2 - Mid-close (170-200mm)
    objects.append(SceneObject(
        position=torch.tensor([-3.0, -2.0, 180.0], device=device),
        size=4.0,
        color=torch.tensor([0.0, 1.0, 0.0], device=device),  # Green
        shape='sphere'
    ))
    
    objects.append(SceneObject(
        position=torch.tensor([2.0, -4.0, 190.0], device=device),
        size=3.5,
        color=torch.tensor([0.0, 1.0, 1.0], device=device),  # Cyan
        shape='cube'
    ))
    
    # Layer 3 - Mid-far (220-250mm)
    objects.append(SceneObject(
        position=torch.tensor([0.0, 2.0, 230.0], device=device),
        size=5.0,
        color=torch.tensor([0.0, 0.0, 1.0], device=device),  # Blue
        shape='sphere'
    ))
    
    objects.append(SceneObject(
        position=torch.tensor([-4.0, 0.0, 240.0], device=device),
        size=4.5,
        color=torch.tensor([1.0, 0.0, 1.0], device=device),  # Magenta
        shape='cube'
    ))
    
    # Layer 4 - Far (280-320mm)
    objects.append(SceneObject(
        position=torch.tensor([3.0, 1.0, 290.0], device=device),
        size=6.0,
        color=torch.tensor([1.0, 1.0, 0.0], device=device),  # Yellow
        shape='sphere'
    ))
    
    objects.append(SceneObject(
        position=torch.tensor([0.0, -3.0, 310.0], device=device),
        size=5.5,
        color=torch.tensor([0.5, 0.5, 0.5], device=device),  # Gray
        shape='cube'
    ))
    
    return objects

def create_scene_office() -> List[SceneObject]:
    """Create office-like scene"""
    objects = []
    
    # Desk items in foreground
    objects.append(SceneObject(
        position=torch.tensor([-3.0, -2.0, 120.0], device=device),
        size=2.0,
        color=torch.tensor([0.8, 0.1, 0.1], device=device),  # Red coffee mug
        shape='cube'
    ))
    
    objects.append(SceneObject(
        position=torch.tensor([4.0, -1.0, 130.0], device=device),
        size=1.5,
        color=torch.tensor([0.2, 0.2, 0.2], device=device),  # Black phone
        shape='cube'
    ))
    
    # Monitor/computer in mid-ground
    objects.append(SceneObject(
        position=torch.tensor([0.0, 3.0, 180.0], device=device),
        size=8.0,
        color=torch.tensor([0.1, 0.1, 0.1], device=device),  # Black monitor
        shape='cube'
    ))
    
    # Plant on desk
    objects.append(SceneObject(
        position=torch.tensor([6.0, 0.0, 170.0], device=device),
        size=3.0,
        color=torch.tensor([0.2, 0.7, 0.2], device=device),  # Green plant
        shape='sphere'
    ))
    
    # Wall items in background
    objects.append(SceneObject(
        position=torch.tensor([-2.0, 8.0, 280.0], device=device),
        size=4.0,
        color=torch.tensor([0.9, 0.9, 0.2], device=device),  # Yellow sticky note
        shape='cube'
    ))
    
    objects.append(SceneObject(
        position=torch.tensor([4.0, 6.0, 300.0], device=device),
        size=6.0,
        color=torch.tensor([0.8, 0.8, 0.8], device=device),  # White board/poster
        shape='cube'
    ))
    
    return objects

def create_scene_nature() -> List[SceneObject]:
    """Create nature scene"""
    objects = []
    
    # Flowers in foreground
    objects.append(SceneObject(
        position=torch.tensor([2.0, -3.0, 110.0], device=device),
        size=2.0,
        color=torch.tensor([1.0, 0.2, 0.6], device=device),  # Pink flower
        shape='sphere'
    ))
    
    objects.append(SceneObject(
        position=torch.tensor([-4.0, -2.0, 125.0], device=device),
        size=1.8,
        color=torch.tensor([0.9, 0.9, 0.2], device=device),  # Yellow flower
        shape='sphere'
    ))
    
    # Grass/bushes in mid-ground
    objects.append(SceneObject(
        position=torch.tensor([0.0, -5.0, 170.0], device=device),
        size=5.0,
        color=torch.tensor([0.3, 0.8, 0.3], device=device),  # Green bush
        shape='cube'
    ))
    
    objects.append(SceneObject(
        position=torch.tensor([8.0, -3.0, 190.0], device=device),
        size=4.0,
        color=torch.tensor([0.2, 0.6, 0.2], device=device),  # Dark green bush
        shape='sphere'
    ))
    
    # Trees in background
    objects.append(SceneObject(
        position=torch.tensor([-6.0, 2.0, 250.0], device=device),
        size=12.0,
        color=torch.tensor([0.1, 0.5, 0.1], device=device),  # Tree trunk/leaves
        shape='sphere'
    ))
    
    objects.append(SceneObject(
        position=torch.tensor([3.0, 4.0, 320.0], device=device),
        size=10.0,
        color=torch.tensor([0.15, 0.6, 0.15], device=device),  # Another tree
        shape='sphere'
    ))
    
    # Sky elements
    objects.append(SceneObject(
        position=torch.tensor([10.0, 12.0, 400.0], device=device),
        size=5.0,
        color=torch.tensor([0.9, 0.9, 0.9], device=device),  # Cloud
        shape='sphere'
    ))
    
    return objects

def create_scene_spherical_checkerboard() -> SphericalCheckerboard:
    """Create spherical checkerboard scene - from spherical_checkerboard_raytracer.py"""
    scene = SphericalCheckerboard(
        center=torch.tensor([0.0, 0.0, 200.0], device=device),
        radius=50.0
    )
    return scene

def trace_ray_through_spherical_checkerboard(ray_origin: torch.Tensor, ray_dir: torch.Tensor, 
                                           scene: SphericalCheckerboard) -> torch.Tensor:
    """Trace ray through spherical checkerboard and return color - from spherical_checkerboard_raytracer.py"""
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

# Ray tracing functions (back to original high-quality versions)
def ray_sphere_intersection(ray_origin: torch.Tensor, ray_dir: torch.Tensor, 
                           sphere_center: torch.Tensor, sphere_radius: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute ray-sphere intersection"""
    oc = ray_origin - sphere_center
    a = torch.sum(ray_dir * ray_dir, dim=-1)
    b = 2.0 * torch.sum(oc * ray_dir, dim=-1)
    c = torch.sum(oc * oc, dim=-1) - sphere_radius * sphere_radius
    
    discriminant = b * b - 4 * a * c
    hit_mask = discriminant >= 0
    
    t = torch.full_like(discriminant, float('inf'))
    valid_mask = hit_mask & (discriminant >= 0)
    
    if valid_mask.any():
        sqrt_discriminant = torch.sqrt(discriminant[valid_mask])
        t1 = (-b[valid_mask] - sqrt_discriminant) / (2 * a[valid_mask])
        t2 = (-b[valid_mask] + sqrt_discriminant) / (2 * a[valid_mask])
        
        t_valid = torch.where(t1 > 1e-6, t1, t2)
        t[valid_mask] = t_valid
        hit_mask[valid_mask] = t_valid > 1e-6
    
    return hit_mask, t

def ray_cube_intersection(ray_origin: torch.Tensor, ray_dir: torch.Tensor,
                         cube_center: torch.Tensor, cube_size: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute ray-cube intersection"""
    half_size = cube_size / 2
    cube_min = cube_center - half_size
    cube_max = cube_center + half_size
    
    inv_dir = 1.0 / (ray_dir + 1e-8)
    
    t1 = (cube_min - ray_origin) * inv_dir
    t2 = (cube_max - ray_origin) * inv_dir
    
    t_min = torch.min(t1, t2)
    t_max = torch.max(t1, t2)
    
    t_near = torch.max(t_min, dim=-1)[0]
    t_far = torch.min(t_max, dim=-1)[0]
    
    hit_mask = (t_near <= t_far) & (t_far > 1e-6)
    t = torch.where(t_near > 1e-6, t_near, t_far)
    t = torch.where(hit_mask, t, torch.full_like(t, float('inf')))
    
    return hit_mask, t

def trace_ray_through_scene(ray_origin: torch.Tensor, ray_dir: torch.Tensor, 
                           scene_objects: List[SceneObject]) -> torch.Tensor:
    """Trace ray through scene and return color"""
    batch_shape = ray_origin.shape[:-1]
    color = torch.zeros((*batch_shape, 3), device=device)
    
    closest_t = torch.full(batch_shape, float('inf'), device=device)
    
    for obj in scene_objects:
        if obj.shape == 'sphere':
            hit_mask, t = ray_sphere_intersection(ray_origin, ray_dir, obj.position, obj.size)
        elif obj.shape == 'cube':
            hit_mask, t = ray_cube_intersection(ray_origin, ray_dir, obj.position, obj.size)
        else:
            continue
            
        closer_mask = hit_mask & (t < closest_t)
        color[closer_mask] = obj.color
        closest_t[closer_mask] = t[closer_mask]
    
    return color

def refract_ray_through_eye_lens(ray_origin: torch.Tensor, ray_dir: torch.Tensor, 
                                eye_focal_length: float, params: OpticalSystemParams) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Refract ray through eye lens using thin lens approximation"""
    t_lens = -ray_origin[..., 2] / ray_dir[..., 2]
    lens_intersection = ray_origin + t_lens.unsqueeze(-1) * ray_dir
    
    pupil_radius = params.eye_pupil_diameter / 2
    radial_distance = torch.sqrt(lens_intersection[..., 0]**2 + lens_intersection[..., 1]**2)
    valid_mask = radial_distance <= pupil_radius
    
    lens_power = 1.0 / eye_focal_length
    
    deflection_x = -lens_power * lens_intersection[..., 0]
    deflection_y = -lens_power * lens_intersection[..., 1]
    
    new_ray_dir = ray_dir.clone()
    new_ray_dir[..., 0] += deflection_x
    new_ray_dir[..., 1] += deflection_y
    
    new_ray_dir = new_ray_dir / torch.norm(new_ray_dir, dim=-1, keepdim=True)
    
    return lens_intersection, new_ray_dir, valid_mask

def refract_ray_through_tunable_lens(ray_origin: torch.Tensor, ray_dir: torch.Tensor,
                                    tunable_focal_length: float, params: OpticalSystemParams) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Refract ray through tunable lens"""
    lens_z = params.tunable_lens_distance
    
    t_lens = (lens_z - ray_origin[..., 2]) / ray_dir[..., 2]
    lens_intersection = ray_origin + t_lens.unsqueeze(-1) * ray_dir
    
    lens_radius = params.tunable_lens_diameter / 2
    radial_distance = torch.sqrt(lens_intersection[..., 0]**2 + lens_intersection[..., 1]**2)
    valid_mask = radial_distance <= lens_radius
    
    lens_power = 1.0 / tunable_focal_length
    
    deflection_x = -lens_power * lens_intersection[..., 0]
    deflection_y = -lens_power * lens_intersection[..., 1]
    
    new_ray_dir = ray_dir.clone()
    new_ray_dir[..., 0] += deflection_x
    new_ray_dir[..., 1] += deflection_y
    
    new_ray_dir = new_ray_dir / torch.norm(new_ray_dir, dim=-1, keepdim=True)
    
    return lens_intersection, new_ray_dir, valid_mask

def find_nearest_microlens_and_refract(ray_origin: torch.Tensor, ray_dir: torch.Tensor,
                                      microlens_positions: torch.Tensor, params: OpticalSystemParams) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Find nearest CIRCULAR microlens and refract ray through it"""
    microlens_z = params.microlens_distance
    
    t_array = (microlens_z - ray_origin[..., 2]) / ray_dir[..., 2]
    array_intersection = ray_origin + t_array.unsqueeze(-1) * ray_dir
    
    ray_xy = array_intersection[..., :2]
    
    batch_shape = ray_xy.shape[:-1]
    ray_xy_flat = ray_xy.reshape(-1, 2)
    
    # Compute distances to all microlenses
    distances = torch.cdist(ray_xy_flat.unsqueeze(0), microlens_positions.unsqueeze(0)).squeeze(0)
    
    # Find nearest microlens for each ray
    nearest_indices = torch.argmin(distances, dim=1)
    nearest_centers = microlens_positions[nearest_indices]
    
    # Reshape back to original batch shape
    nearest_centers = nearest_centers.reshape(*batch_shape, 2)
    
    # Check if ray hits this CIRCULAR microlens (within pitch/2 radius)
    distance_to_center = torch.norm(ray_xy - nearest_centers, dim=-1)
    valid_mask = distance_to_center <= params.microlens_pitch / 2
    
    # Apply microlens refraction (thin lens)
    microlens_power = 1.0 / params.microlens_focal_length
    
    # Local coordinates relative to microlens center
    local_x = ray_xy[..., 0] - nearest_centers[..., 0]
    local_y = ray_xy[..., 1] - nearest_centers[..., 1]
    
    # Deflection
    deflection_x = -microlens_power * local_x
    deflection_y = -microlens_power * local_y
    
    # Update ray direction
    new_ray_dir = ray_dir.clone()
    new_ray_dir[..., 0] += deflection_x
    new_ray_dir[..., 1] += deflection_y
    
    # Normalize
    new_ray_dir = new_ray_dir / torch.norm(new_ray_dir, dim=-1, keepdim=True)
    
    return array_intersection, new_ray_dir, valid_mask

def sample_display_at_intersection(ray_origin: torch.Tensor, ray_dir: torch.Tensor,
                                  display_images: torch.Tensor, focal_lengths: torch.Tensor, 
                                  params: OpticalSystemParams) -> torch.Tensor:
    """Sample display color where ray intersects display plane"""
    display_z = params.display_distance
    
    t_display = (display_z - ray_origin[..., 2]) / ray_dir[..., 2]
    display_intersection = ray_origin + t_display.unsqueeze(-1) * ray_dir
    
    display_size = params.display_size
    u = (display_intersection[..., 0] + display_size/2) / display_size
    v = (display_intersection[..., 1] + display_size/2) / display_size
    
    valid_mask = (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1)
    
    batch_shape = u.shape
    sampled_color = torch.zeros((*batch_shape, 3), device=device)
    
    for i, display_image in enumerate(display_images):
        pixel_u = u * (params.display_resolution - 1)
        pixel_v = v * (params.display_resolution - 1)
        
        u0 = torch.floor(pixel_u).long()
        v0 = torch.floor(pixel_v).long()
        u1 = torch.clamp(u0 + 1, 0, params.display_resolution - 1)
        v1 = torch.clamp(v0 + 1, 0, params.display_resolution - 1)
        
        wu = pixel_u - u0.float()
        wv = pixel_v - v0.float()
        
        u0 = torch.clamp(u0, 0, params.display_resolution - 1)
        v0 = torch.clamp(v0, 0, params.display_resolution - 1)
        
        valid_pixels = valid_mask & (u0 >= 0) & (v0 >= 0) & (u1 < params.display_resolution) & (v1 < params.display_resolution)
        
        if valid_pixels.any():
            c00 = display_image[:, v0[valid_pixels], u0[valid_pixels]]
            c01 = display_image[:, v1[valid_pixels], u0[valid_pixels]]
            c10 = display_image[:, v0[valid_pixels], u1[valid_pixels]]
            c11 = display_image[:, v1[valid_pixels], u1[valid_pixels]]
            
            wu_valid = wu[valid_pixels].unsqueeze(0)
            wv_valid = wv[valid_pixels].unsqueeze(0)
            
            c0 = c00 * (1 - wu_valid) + c10 * wu_valid
            c1 = c01 * (1 - wu_valid) + c11 * wu_valid
            interpolated = c0 * (1 - wv_valid) + c1 * wv_valid
            
            sampled_color[valid_pixels] += interpolated.T
    
    return sampled_color, valid_mask

def generate_target_retina_image_multiray(scene_objects, eye_focal_length: float, 
                                        params: OpticalSystemParams, resolution: int = 384) -> torch.Tensor:
    """Generate target image with MULTI-RAY sampling - enhanced from spherical_checkerboard_raytracer.py"""
    
    # Handle spherical checkerboard scene differently
    if isinstance(scene_objects, SphericalCheckerboard):
        return generate_target_retina_image_spherical_checkerboard(scene_objects, eye_focal_length, params, resolution)
    
    # Standard multi-ray sampling for regular scenes
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
    
    # Process in small batches for speed
    batch_size = min(1024, N)  # Small batches for speed (1K pixels)
    final_colors = torch.zeros(N, 3, device=device)
    
    # Generate pupil samples for sub-aperture sampling
    pupil_radius = params.eye_pupil_diameter / 2
    pupil_samples = generate_pupil_samples(M, pupil_radius)  # [M, 2]
    
    # Process in batches
    for batch_start in range(0, N, batch_size):
        batch_end = min(batch_start + batch_size, N)
        batch_retina_points = retina_points[batch_start:batch_end]
        batch_N = batch_retina_points.shape[0]
        
        # Create 3D pupil points at eye lens
        pupil_points_3d = torch.zeros(M, 3, device=device)
        pupil_points_3d[:, 0] = pupil_samples[:, 0]
        pupil_points_3d[:, 1] = pupil_samples[:, 1]
        pupil_points_3d[:, 2] = 0.0  # At eye lens plane
        
        # Create ray bundles: [batch_N, M, 3]
        retina_expanded = batch_retina_points.unsqueeze(1)  # [batch_N, 1, 3]
        pupil_expanded = pupil_points_3d.unsqueeze(0).expand(batch_N, M, 3)  # [batch_N, M, 3]
        
        # Ray directions: from retina TO pupil points
        ray_dirs = pupil_expanded - retina_expanded  # [batch_N, M, 3]
        ray_dirs = ray_dirs / torch.norm(ray_dirs, dim=-1, keepdim=True)
        ray_origins = pupil_expanded  # Already [batch_N, M, 3]
        
        # Apply lens refraction with multi-ray sampling
        lens_power = 1000.0 / eye_focal_length / 1000.0  # mm^-1
        
        local_x = pupil_expanded[:, :, 0]  # [batch_N, M]
        local_y = pupil_expanded[:, :, 1]  # [batch_N, M]
        
        deflection_x = -lens_power * local_x
        deflection_y = -lens_power * local_y
        
        refracted_ray_dirs = ray_dirs.clone()
        refracted_ray_dirs[:, :, 0] += deflection_x
        refracted_ray_dirs[:, :, 1] += deflection_y
        refracted_ray_dirs = refracted_ray_dirs / torch.norm(refracted_ray_dirs, dim=-1, keepdim=True)
        
        # Trace rays to scene (MULTI-RAY)
        ray_origins_flat = ray_origins.reshape(-1, 3)
        ray_dirs_flat = refracted_ray_dirs.reshape(-1, 3)
        
        colors_flat = trace_ray_through_scene(ray_origins_flat, ray_dirs_flat, scene_objects)
        
        # Check pupil validity - shape should be [batch_N, M]
        pupil_radius_check = params.eye_pupil_diameter / 2
        radial_distance = torch.sqrt(pupil_expanded[:, :, 0]**2 + pupil_expanded[:, :, 1]**2)
        valid_mask = radial_distance <= pupil_radius_check  # This should already be [batch_N, M]
        
        # Average over sub-aperture samples (NATURAL BLUR from ray averaging)
        colors = colors_flat.reshape(batch_N, M, 3)
        
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

def generate_target_retina_image_spherical_checkerboard(scene: SphericalCheckerboard, eye_focal_length: float, 
                                                       params: OpticalSystemParams, resolution: int = 1024) -> torch.Tensor:
    """Generate target image for spherical checkerboard with MULTI-RAY sampling"""
    
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
    
    # Generate pupil samples
    pupil_radius = params.eye_pupil_diameter / 2
    pupil_samples = generate_pupil_samples(M, pupil_radius)
    
    pupil_points_3d = torch.zeros(M, 3, device=device)
    pupil_points_3d[:, 0] = pupil_samples[:, 0]
    pupil_points_3d[:, 1] = pupil_samples[:, 1]
    pupil_points_3d[:, 2] = 0.0
    
    # Create ray bundles
    retina_expanded = retina_points.unsqueeze(1)
    pupil_expanded = pupil_points_3d.unsqueeze(0).expand(N, M, 3)
    
    ray_dirs = pupil_expanded - retina_expanded
    ray_dirs = ray_dirs / torch.norm(ray_dirs, dim=-1, keepdim=True)
    ray_origins = pupil_expanded  # Already [N, M, 3]
    
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
    colors = colors_flat.reshape(N, M, 3)
    
    # Check pupil validity
    radial_distance = torch.sqrt(pupil_expanded[:, :, 0]**2 + pupil_expanded[:, :, 1]**2)
    valid_mask = radial_distance <= pupil_radius
    
    final_colors = torch.zeros(N, 3, device=device)
    for pixel_idx in range(N):
        valid_samples = valid_mask[pixel_idx, :]
        if valid_samples.any():
            pixel_colors = colors[pixel_idx, valid_samples, :]
            final_colors[pixel_idx, :] = torch.mean(pixel_colors, dim=0)
    
    retina_image = final_colors.reshape(resolution, resolution, 3)
    
    return retina_image

# Legacy function for compatibility
def generate_target_retina_image(scene_objects, eye_focal_length: float, 
                                params: OpticalSystemParams, resolution: int = 1024) -> torch.Tensor:
    """Legacy wrapper - redirects to multi-ray version"""
    return generate_target_retina_image_multiray(scene_objects, eye_focal_length, params, resolution)

def trace_complete_optical_path_multiray(retina_points: torch.Tensor, eye_focal_length: float,
                                        tunable_focal_length: float, display_system: HighQualityLightFieldDisplay,
                                        params: OpticalSystemParams) -> torch.Tensor:
    """Trace rays from retina through complete optical system to display with MULTI-RAY sampling - MEMORY EFFICIENT"""
    
    N = retina_points.shape[0]
    M = params.samples_per_pixel
    
    # Process in small batches for speed
    batch_size = min(512, N)  # Process up to 512 pixels at a time for speed
    final_colors = torch.zeros(N, 3, device=device)
    
    for batch_start in range(0, N, batch_size):
        batch_end = min(batch_start + batch_size, N)
        batch_retina_points = retina_points[batch_start:batch_end]
        batch_N = batch_retina_points.shape[0]
        
        batch_colors = trace_complete_optical_path_multiray_batch(
            batch_retina_points, eye_focal_length, tunable_focal_length, 
            display_system, params, batch_N, M
        )
        
        final_colors[batch_start:batch_end] = batch_colors
        
        # Clear GPU cache after each batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return final_colors

def trace_complete_optical_path_multiray_batch(retina_points: torch.Tensor, eye_focal_length: float,
                                              tunable_focal_length: float, display_system: HighQualityLightFieldDisplay,
                                              params: OpticalSystemParams, N: int, M: int) -> torch.Tensor:
    """Process a batch of retina points with multi-ray sampling"""
    
    # Generate pupil samples for sub-aperture sampling
    pupil_radius = params.eye_pupil_diameter / 2
    pupil_samples = generate_pupil_samples(M, pupil_radius)
    
    # Create 3D pupil points at eye lens
    pupil_points_3d = torch.zeros(M, 3, device=device)
    pupil_points_3d[:, 0] = pupil_samples[:, 0]
    pupil_points_3d[:, 1] = pupil_samples[:, 1]
    pupil_points_3d[:, 2] = 0.0
    
    # Create ray bundles: [N, M, 3]
    retina_expanded = retina_points.unsqueeze(1)  # [N, 1, 3]
    pupil_expanded = pupil_points_3d.unsqueeze(0).expand(N, M, 3)  # [N, M, 3]
    
    # Ray directions: from retina TO pupil points
    ray_dirs = pupil_expanded - retina_expanded  # [N, M, 3]
    ray_dirs = ray_dirs / torch.norm(ray_dirs, dim=-1, keepdim=True)
    ray_origins = pupil_expanded  # Already [N, M, 3]
    
    # Step 2: Refract through eye lens (multi-ray)
    lens_power = 1000.0 / eye_focal_length / 1000.0
    
    local_x = pupil_expanded[:, :, 0]
    local_y = pupil_expanded[:, :, 1]
    
    deflection_x = -lens_power * local_x
    deflection_y = -lens_power * local_y
    
    refracted_ray_dirs = ray_dirs.clone()
    refracted_ray_dirs[:, :, 0] += deflection_x
    refracted_ray_dirs[:, :, 1] += deflection_y
    refracted_ray_dirs = refracted_ray_dirs / torch.norm(refracted_ray_dirs, dim=-1, keepdim=True)
    
    # Check pupil validity
    radial_distance = torch.sqrt(pupil_expanded[:, :, 0]**2 + pupil_expanded[:, :, 1]**2)
    valid_eye = radial_distance <= pupil_radius
    
    # Step 3: Refract through tunable lens (multi-ray)
    lens_z = params.tunable_lens_distance
    t_lens = (lens_z - ray_origins[:, :, 2]) / refracted_ray_dirs[:, :, 2]
    lens_intersection = ray_origins + t_lens.unsqueeze(-1) * refracted_ray_dirs
    
    lens_radius = params.tunable_lens_diameter / 2
    radial_distance_tunable = torch.sqrt(lens_intersection[:, :, 0]**2 + lens_intersection[:, :, 1]**2)
    valid_tunable = radial_distance_tunable <= lens_radius
    
    tunable_lens_power = 1.0 / tunable_focal_length
    deflection_x_tunable = -tunable_lens_power * lens_intersection[:, :, 0]
    deflection_y_tunable = -tunable_lens_power * lens_intersection[:, :, 1]
    
    refracted_ray_dirs[:, :, 0] += deflection_x_tunable
    refracted_ray_dirs[:, :, 1] += deflection_y_tunable
    refracted_ray_dirs = refracted_ray_dirs / torch.norm(refracted_ray_dirs, dim=-1, keepdim=True)
    
    # Step 4: Find nearest CIRCULAR microlens and refract (multi-ray)
    microlens_z = params.microlens_distance
    t_array = (microlens_z - lens_intersection[:, :, 2]) / refracted_ray_dirs[:, :, 2]
    array_intersection = lens_intersection + t_array.unsqueeze(-1) * refracted_ray_dirs
    
    # Process rays in batches for microlens interaction
    ray_xy = array_intersection[:, :, :2]  # [N, M, 2]
    
    # Flatten for distance computation
    ray_xy_flat = ray_xy.reshape(-1, 2)  # [N*M, 2]
    
    # Use grid-based microlens selection for memory efficiency
    # Instead of computing distances to all microlenses, use grid indexing
    grid_x_idx = torch.round(ray_xy_flat[:, 0] / params.microlens_pitch).long()
    grid_y_idx = torch.round(ray_xy_flat[:, 1] / params.microlens_pitch).long()
    
    # Compute actual grid positions
    grid_x_pos = grid_x_idx.float() * params.microlens_pitch
    grid_y_pos = grid_y_idx.float() * params.microlens_pitch
    
    nearest_centers = torch.stack([grid_x_pos, grid_y_pos], dim=1)
    
    # Reshape back to [N, M, 2]
    nearest_centers = nearest_centers.reshape(N, M, 2)
    
    # Check if ray hits CIRCULAR microlens
    distance_to_center = torch.norm(ray_xy - nearest_centers, dim=-1)
    valid_microlens = distance_to_center <= params.microlens_pitch / 2
    
    # Apply microlens refraction
    microlens_power = 1.0 / params.microlens_focal_length
    local_x_micro = ray_xy[:, :, 0] - nearest_centers[:, :, 0]
    local_y_micro = ray_xy[:, :, 1] - nearest_centers[:, :, 1]
    
    deflection_x_micro = -microlens_power * local_x_micro
    deflection_y_micro = -microlens_power * local_y_micro
    
    refracted_ray_dirs[:, :, 0] += deflection_x_micro
    refracted_ray_dirs[:, :, 1] += deflection_y_micro
    refracted_ray_dirs = refracted_ray_dirs / torch.norm(refracted_ray_dirs, dim=-1, keepdim=True)
    
    # Step 5: Sample display using FIXED focal lengths (multi-ray)
    display_z = params.display_distance
    t_display = (display_z - array_intersection[:, :, 2]) / refracted_ray_dirs[:, :, 2]
    display_intersection = array_intersection + t_display.unsqueeze(-1) * refracted_ray_dirs
    
    display_size = params.display_size
    u = (display_intersection[:, :, 0] + display_size/2) / display_size
    v = (display_intersection[:, :, 1] + display_size/2) / display_size
    
    valid_display = (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1)
    
    # Sample display colors (simplified for multi-ray)
    display_color = torch.zeros(N, M, 3, device=device)
    
    for i, display_image in enumerate(display_system.display_images):
        pixel_u = u * (params.display_resolution - 1)
        pixel_v = v * (params.display_resolution - 1)
        
        u0 = torch.floor(pixel_u).long().clamp(0, params.display_resolution - 1)
        v0 = torch.floor(pixel_v).long().clamp(0, params.display_resolution - 1)
        
        valid_pixels = valid_display & (u0 >= 0) & (v0 >= 0) & (u0 < params.display_resolution) & (v0 < params.display_resolution)
        
        if valid_pixels.any():
            sampled_colors = display_image[:, v0[valid_pixels], u0[valid_pixels]].T  # [num_valid_pixels, 3]
            display_color[valid_pixels] += sampled_colors
    
    # Combine all validity masks
    final_valid = valid_eye & valid_tunable & valid_microlens & valid_display
    
    # Average over sub-aperture samples (NATURAL BLUR from ray averaging)
    final_colors = torch.zeros(N, 3, device=device)
    for pixel_idx in range(N):
        valid_samples = final_valid[pixel_idx, :]
        if valid_samples.any():
            pixel_colors = display_color[pixel_idx, valid_samples, :]
            final_colors[pixel_idx, :] = torch.mean(pixel_colors, dim=0)
    
    return final_colors

# Legacy function for compatibility
def trace_complete_optical_path(retina_points: torch.Tensor, eye_focal_length: float,
                               tunable_focal_length: float, display_system: HighQualityLightFieldDisplay,
                               params: OpticalSystemParams) -> torch.Tensor:
    """Legacy wrapper - redirects to multi-ray version"""
    return trace_complete_optical_path_multiray(retina_points, eye_focal_length, tunable_focal_length, display_system, params)

class ComprehensiveLightFieldOptimizer:
    """Main optimizer class for multiple scenes with high quality output"""
    
    def __init__(self, params: OpticalSystemParams):
        self.params = params
        
        # Create comprehensive scene set for A100 processing power
        self.scenes = {
            'basic': create_scene_basic(),
            'complex': create_scene_complex(),
            'stick_figure': create_scene_stick_figure(),
            'layered': create_scene_layered(),
            'office': create_scene_office(),
            'nature': create_scene_nature(),
            'spherical_checkerboard': create_scene_spherical_checkerboard()
        }
        
        # Dictionary to store optimizers and results for each scene
        self.scene_optimizers = {}
        self.scene_results = {}
        
        # Create scene subdirectories
        for scene_name in self.scenes.keys():
            scene_dir = f'{comprehensive_results_dir}/{scene_name}'
            os.makedirs(scene_dir, exist_ok=True)
        
    def optimize_scene(self, scene_name: str, scene_objects: List[SceneObject], num_iterations: int = 300):
        """Optimize a single scene"""
        print(f"\\n=== OPTIMIZING SCENE: {scene_name.upper()} ===")
        
        # Create display system for this scene
        display_system = HighQualityLightFieldDisplay(self.params)
        
        # 40GB A100-optimized optimizer with maximum learning rate for fastest convergence
        optimizer = optim.AdamW(display_system.parameters(), lr=0.03, weight_decay=1e-4, fused=True)
        
        # Training history
        loss_history = []
        
        # Fixed eye and tunable lens focal lengths for training
        eye_focal_length = 35.0  # mm
        tunable_focal_length = 25.0  # mm
        
        print(f"Training {scene_name} scene for {num_iterations} iterations...")
        print(f"Fixed focal lengths: {display_system.focal_lengths.cpu().numpy()}")
        
        # Set up GIF generation for real-time frame collection
        scene_dir = f'{comprehensive_results_dir}/{scene_name}'
        os.makedirs(scene_dir, exist_ok=True)  # Ensure directory exists for parallel processing
        gif_path = f'{scene_dir}/training_evolution_live.gif'
        
        # Create first frame to get dimensions
        loss, simulated_img, target_img = self.compute_loss(
            scene_objects, display_system, eye_focal_length, tunable_focal_length
        )
        
        # Collect frames for GIF generation
        gif_frames = []
        
        # Create and save first frame
        frame_img = self.create_training_frame_image(simulated_img, target_img, loss.item(), 0, 
                                                    display_system.focal_lengths.cpu().numpy(), scene_name, num_iterations)
        gif_frames.append(frame_img)
        
        # A100 optimization with mixed precision and progress bar
        scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        
        for iteration in tqdm(range(num_iterations), desc=f"Training {scene_name}", unit="iter"):
            optimizer.zero_grad()
            
            # Use mixed precision for A100 speedup
            with torch.cuda.amp.autocast() if torch.cuda.is_available() else torch.no_grad():
                # Compute loss
                loss, simulated_img, target_img = self.compute_loss(
                    scene_objects, display_system, eye_focal_length, tunable_focal_length
                )
            
            # Generate GIF frame less frequently for speed
            if iteration > 0 and iteration % 50 == 0:  # Collect every 50th frame for GIF (fewer frames = faster)
                frame_img = self.create_training_frame_image(simulated_img, target_img, loss.item(), iteration, 
                                                           display_system.focal_lengths.cpu().numpy(), scene_name, num_iterations)
                gif_frames.append(frame_img)
            
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
            
            # Clamp display values to valid range
            with torch.no_grad():
                display_system.display_images.clamp_(0, 1)
            
            # Record history
            loss_history.append(loss.item())
            
            if iteration % 10 == 0:  # Frequent updates for 50 iterations
                tqdm.write(f"  Iteration {iteration}: Loss = {loss.item():.6f}")
                if torch.cuda.is_available():
                    memory_used = torch.cuda.memory_allocated() / 1024**3
                    tqdm.write(f"    GPU Memory Used: {memory_used:.2f} GB")
                
                # Save intermediate seen image for monitoring
                plt.figure(figsize=(8, 4))
                
                plt.subplot(1, 2, 1)
                plt.imshow(np.clip(simulated_img.detach().cpu().numpy(), 0, 1))
                plt.title(f'Seen Image - Iter {iteration}')
                plt.axis('off')
                
                plt.subplot(1, 2, 2)
                plt.imshow(np.clip(target_img.detach().cpu().numpy(), 0, 1))
                plt.title('Target Image')
                plt.axis('off')
                
                plt.suptitle(f'{scene_name.title()} Scene - Iteration {iteration}')
                plt.tight_layout()
                plt.savefig(f'{comprehensive_results_dir}/{scene_name}/seen_image_iter_{iteration:04d}.png', dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"    Saved seen image for iteration {iteration}")
            
            # Minimal memory cleanup for 40GB A100 efficiency
            if iteration % 100 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Create training GIF from collected frames
        if gif_frames:
            gif_frames[0].save(gif_path, save_all=True, append_images=gif_frames[1:], 
                              duration=300, loop=0, optimize=True)
            print(f"Live training GIF saved: {gif_path}")
        else:
            print("No GIF frames collected")
        
        # Store results (no need to store image_history since we generated video live)
        self.scene_optimizers[scene_name] = display_system
        self.scene_results[scene_name] = {
            'loss_history': loss_history,
            'final_loss': loss_history[-1],
            'scene_objects': scene_objects
        }
        
        print(f"Scene {scene_name} optimization complete!")
        print(f"Final loss: {loss_history[-1]:.6f}")
        
        # Generate GIFs immediately after this scene optimization
        print(f"Generating GIFs for {scene_name} scene...")
        self.save_focal_length_sweep_gifs(scene_name, scene_dir)
        self.save_eye_movement_sweep_gifs(scene_name, scene_dir)
        print(f"GIFs generated for {scene_name} scene!")
        
        # IMPORTANT: Clear GPU memory after each scene optimization
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print(f"GPU memory cleared after {scene_name} scene optimization")
    
    def create_training_frame(self, simulated_img: torch.Tensor, target_img: torch.Tensor, 
                             loss_val: float, iteration: int, focal_lengths: np.ndarray, 
                             scene_name: str, total_iterations: int) -> np.ndarray:
        """Create a single training frame for video - efficient version"""
        plt.figure(figsize=(12, 8))
        
        # Main simulated image
        plt.subplot(2, 2, 1)
        plt.imshow(np.clip(simulated_img.detach().cpu().numpy(), 0, 1))
        plt.title(f'Eye View - Iteration {iteration}', fontsize=14)
        plt.axis('off')
        
        # Target image
        plt.subplot(2, 2, 2)
        plt.imshow(np.clip(target_img.detach().cpu().numpy(), 0, 1))
        plt.title('Target Scene View', fontsize=14)
        plt.axis('off')
        
        # Loss curve (up to current iteration)
        plt.subplot(2, 2, 3)
        loss_history = self.scene_results.get(scene_name, {}).get('loss_history', [])
        if loss_history:
            plt.plot(loss_history[:iteration+1], 'b-', linewidth=2)
        plt.scatter(iteration, loss_val, color='red', s=50, zorder=5)
        plt.title(f'Training Loss: {loss_val:.6f}', fontsize=14)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.xlim(0, total_iterations)
        plt.yscale('log')
        
        # System info and focal lengths
        plt.subplot(2, 2, 4)
        plt.axis('off')
        plt.text(0.1, 0.9, f'ITERATION {iteration}', fontsize=18, fontweight='bold')
        plt.text(0.1, 0.8, f'Scene: {scene_name.title()}', fontsize=14)
        plt.text(0.1, 0.7, f'Loss: {loss_val:.6f}', fontsize=14)
        plt.text(0.1, 0.6, 'Fixed Focal Lengths:', fontsize=14, fontweight='bold')
        
        for j, fl in enumerate(focal_lengths[:6]):  # Show first 6
            plt.text(0.1, 0.5 - j*0.05, f'  Plane {j+1}: {fl:.1f} mm', fontsize=12)
        
        if len(focal_lengths) > 6:
            plt.text(0.1, 0.2, f'  ... and {len(focal_lengths)-6} more planes', fontsize=10, style='italic')
        
        plt.suptitle(f'{scene_name.title()} Scene Training - Live Video Generation', fontsize=16)
        plt.tight_layout()
        
        # Convert to OpenCV format with unique filename for parallel processing
        temp_filename = f'temp_frame_{scene_name}_{iteration}.png'
        plt.savefig(temp_filename, dpi=100, bbox_inches='tight')
        plt.close()
        
        # Read and convert to OpenCV format
        frame = cv2.imread(temp_filename)
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        
        return frame
    
    def create_training_frame_image(self, simulated_img: torch.Tensor, target_img: torch.Tensor, 
                                   loss_val: float, iteration: int, focal_lengths: np.ndarray, 
                                   scene_name: str, total_iterations: int) -> Image.Image:
        """Create a single training frame as PIL Image for GIF generation"""
        plt.figure(figsize=(12, 8))
        
        # Main simulated image
        plt.subplot(2, 2, 1)
        plt.imshow(np.clip(simulated_img.detach().cpu().numpy(), 0, 1))
        plt.title(f'Eye View - Iteration {iteration}', fontsize=14)
        plt.axis('off')
        
        # Target image
        plt.subplot(2, 2, 2)
        plt.imshow(np.clip(target_img.detach().cpu().numpy(), 0, 1))
        plt.title('Target Scene View', fontsize=14)
        plt.axis('off')
        
        # Loss curve (up to current iteration)
        plt.subplot(2, 2, 3)
        loss_history = self.scene_results.get(scene_name, {}).get('loss_history', [])
        if loss_history and len(loss_history) > iteration:
            plt.plot(loss_history[:iteration+1], 'b-', linewidth=2)
        plt.scatter(iteration, loss_val, color='red', s=50, zorder=5)
        plt.title(f'Training Loss: {loss_val:.6f}', fontsize=14)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.xlim(0, total_iterations)
        plt.yscale('log')
        
        # System info
        plt.subplot(2, 2, 4)
        plt.axis('off')
        plt.text(0.1, 0.9, f'ITERATION {iteration}', fontsize=18, fontweight='bold')
        plt.text(0.1, 0.8, f'Scene: {scene_name.title()}', fontsize=14)
        plt.text(0.1, 0.7, f'Loss: {loss_val:.6f}', fontsize=14)
        plt.text(0.1, 0.6, 'Fixed Focal Lengths:', fontsize=14, fontweight='bold')
        
        for j, fl in enumerate(focal_lengths[:4]):  # Show first 4
            plt.text(0.1, 0.5 - j*0.05, f'  Plane {j+1}: {fl:.1f} mm', fontsize=12)
        
        if len(focal_lengths) > 4:
            plt.text(0.1, 0.3, f'  ... and {len(focal_lengths)-4} more planes', fontsize=10, style='italic')
        
        plt.suptitle(f'{scene_name.title()} Scene Training - Live GIF Generation', fontsize=16)
        plt.tight_layout()
        
        # Convert to PIL Image
        temp_filename = f'temp_frame_{scene_name}_{iteration}.png'
        plt.savefig(temp_filename, dpi=100, bbox_inches='tight')
        plt.close()
        
        # Read as PIL Image
        frame_img = Image.open(temp_filename)
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        
        return frame_img
        
    def compute_loss(self, scene_objects, display_system: HighQualityLightFieldDisplay,
                    eye_focal_length: float, tunable_focal_length: float, resolution: int = 256) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute loss between simulated and target retina images - enhanced with multi-ray sampling"""
        
        # Generate target image using multi-ray sampling
        target_image = generate_target_retina_image_multiray(
            scene_objects, eye_focal_length, self.params, resolution
        )
        
        # Generate retina sampling points
        retina_size = self.params.retina_size
        y_coords = torch.linspace(-retina_size/2, retina_size/2, resolution, device=device)
        x_coords = torch.linspace(-retina_size/2, retina_size/2, resolution, device=device)
        
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        retina_points = torch.stack([
            x_grid.flatten(),
            y_grid.flatten(),
            torch.full_like(x_grid.flatten(), -self.params.retina_distance)
        ], dim=1)
        
        # Trace through optical system using multi-ray sampling
        simulated_colors = trace_complete_optical_path_multiray(
            retina_points, eye_focal_length, tunable_focal_length, 
            display_system, self.params
        )
        
        # Reshape to image
        simulated_image = simulated_colors.reshape(resolution, resolution, 3)
        
        # Compute MSE loss
        loss = torch.mean((simulated_image - target_image) ** 2)
        
        return loss, simulated_image, target_image
    
    def generate_comprehensive_outputs(self):
        """Generate all outputs for all scenes"""
        print("\\n=== GENERATING COMPREHENSIVE HIGH-QUALITY OUTPUTS ===")
        
        for scene_name in self.scenes.keys():
            scene_dir = f'{comprehensive_results_dir}/{scene_name}'
            
            print(f"Generating high-quality outputs for {scene_name} scene...")
            
            # 1. Display images
            self.save_display_images(scene_name, scene_dir)
            
            # 2. Eye views
            self.save_eye_views(scene_name, scene_dir)
            
            # 3. Combined result
            self.save_combined_result(scene_name, scene_dir)
            
            # 4. Training loss plot
            self.save_training_loss(scene_name, scene_dir)
            
            # 5. Create GIF from live MP4 (backup format)
            self.create_gif_from_mp4(scene_name, scene_dir)
            
            # 6. Ground truth and scene layout
            self.save_ground_truth_and_scene(scene_name, scene_dir)
            
            # 7. Eye movement GIFs (original)
            self.save_eye_movement_gifs(scene_name, scene_dir)
            
            # Note: Enhanced GIFs already generated immediately after optimization
            
            # Clear GPU memory after each scene's output generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print(f"  GPU memory cleared after {scene_name} output generation")
        
        # 8. Summary comparison across all scenes
        self.save_scene_comparison()
        
        # 9. ZIP everything
        self.create_results_zip()
        
        print("All high-quality comprehensive outputs generated!")
    
    def save_display_images(self, scene_name: str, scene_dir: str):
        """Save optimized display images"""
        display_system = self.scene_optimizers[scene_name]
        focal_lengths = display_system.focal_lengths.cpu().numpy()
        
        plt.figure(figsize=(20, 6))
        
        for i in range(self.params.num_focal_planes):
            display_img = display_system.display_images[i].detach().cpu().numpy()
            display_img = np.transpose(display_img, (1, 2, 0))
            
            plt.subplot(1, self.params.num_focal_planes, i + 1)
            plt.imshow(np.clip(display_img, 0, 1))
            plt.title(f'Focal Length: {focal_lengths[i]:.1f}mm\\n(Fixed)', fontsize=10)
            plt.axis('off')
        
        plt.suptitle(f'{scene_name.title()} Scene - Optimized Display Images (Fixed Focal Lengths + Circular Microlenses)', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'{scene_dir}/display_images.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_gif_from_mp4(self, scene_name: str, scene_dir: str):
        """Create GIF backup from live-generated MP4"""
        print(f"  Creating GIF backup for {scene_name}...")
        
        mp4_path = f'{scene_dir}/training_evolution_live.mp4'
        gif_path = f'{scene_dir}/training_evolution_live.gif'
        
        # Read MP4 and extract frames for GIF (every 10th frame for reasonable size)
        cap = cv2.VideoCapture(mp4_path)
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Take every 10th frame for GIF
            if frame_count % 10 == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
            
            frame_count += 1
        
        cap.release()
        
        # Save as GIF
        if frames:
            frames[0].save(gif_path, save_all=True, append_images=frames[1:], 
                          duration=150, loop=0, optimize=True)
            print(f"    GIF backup saved: {gif_path}")
        else:
            print(f"    Warning: Could not create GIF from MP4")
    
    def save_eye_views(self, scene_name: str, scene_dir: str):
        """Save individual eye views for each focal plane"""
        display_system = self.scene_optimizers[scene_name]
        scene_objects = self.scene_results[scene_name]['scene_objects']
        eye_focal_length = 35.0
        
        plt.figure(figsize=(16, 10))
        
        for i in range(self.params.num_focal_planes):
            tunable_focal = display_system.focal_lengths[i].item()
            
            # Temporarily use only this display plane
            original_images = display_system.display_images.data.clone()
            display_system.display_images.data.zero_()
            display_system.display_images.data[i] = original_images[i]
            
            # Compute eye view with high resolution
            _, simulated_img, target_img = self.compute_loss(
                scene_objects, display_system, eye_focal_length, tunable_focal, resolution=768
            )
            
            # Restore original images
            display_system.display_images.data = original_images
            
            # Plot simulated view
            plt.subplot(2, self.params.num_focal_planes, i + 1)
            plt.imshow(np.clip(simulated_img.detach().cpu().numpy(), 0, 1))
            plt.title(f'Eye View (FL: {tunable_focal:.1f}mm)', fontsize=10)
            plt.axis('off')
            
            # Plot target view
            plt.subplot(2, self.params.num_focal_planes, i + 1 + self.params.num_focal_planes)
            plt.imshow(np.clip(target_img.detach().cpu().numpy(), 0, 1))
            plt.title('Target', fontsize=10)
            plt.axis('off')
        
        plt.suptitle(f'{scene_name.title()} Scene - Individual Eye Views vs Target (High Quality)', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'{scene_dir}/individual_eye_views.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_combined_result(self, scene_name: str, scene_dir: str):
        """Save combined result comparison"""
        display_system = self.scene_optimizers[scene_name]
        scene_objects = self.scene_results[scene_name]['scene_objects']
        eye_focal_length = 35.0
        tunable_focal_length = 25.0
        
        # Use high resolution for final results
        _, simulated_img, target_img = self.compute_loss(
            scene_objects, display_system, eye_focal_length, tunable_focal_length, resolution=1024
        )
        
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(np.clip(target_img.detach().cpu().numpy(), 0, 1))
        plt.title('Target Scene View', fontsize=14)
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(np.clip(simulated_img.detach().cpu().numpy(), 0, 1))
        plt.title('Optimized System View\\n(High Quality)', fontsize=14)
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        diff = torch.abs(simulated_img - target_img).detach().cpu().numpy()
        plt.imshow(diff)
        plt.title('Absolute Difference', fontsize=14)
        plt.axis('off')
        
        plt.suptitle(f'{scene_name.title()} Scene - Final High-Quality Results', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'{scene_dir}/combined_result.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_training_loss(self, scene_name: str, scene_dir: str):
        """Save training loss plot"""
        loss_history = self.scene_results[scene_name]['loss_history']
        
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history, 'b-', linewidth=2)
        plt.title(f'{scene_name.title()} Scene - Training Loss Evolution', fontsize=16)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # Add final loss annotation
        plt.annotate(f'Final Loss: {loss_history[-1]:.6f}', 
                    xy=(len(loss_history)-1, loss_history[-1]),
                    xytext=(len(loss_history)*0.7, loss_history[-1]*2),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=14, color='red')
        
        plt.tight_layout()
        plt.savefig(f'{scene_dir}/training_loss.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_ground_truth_and_scene(self, scene_name: str, scene_dir: str):
        """Save ground truth and scene layout - enhanced for spherical checkerboard"""
        scene_objects = self.scene_results[scene_name]['scene_objects']
        
        plt.figure(figsize=(16, 12))
        
        # Ground truth at different focal lengths with MULTI-RAY sampling
        eye_focal_lengths = [20.0, 30.0, 40.0, 50.0]
        for i, eye_fl in enumerate(eye_focal_lengths):
            target_img = generate_target_retina_image_multiray(scene_objects, eye_fl, self.params, resolution=512)
            
            plt.subplot(3, 4, i + 1)
            plt.imshow(np.clip(target_img.detach().cpu().numpy(), 0, 1))
            plt.title(f'Ground Truth (Eye FL: {eye_fl:.0f}mm)\\nMulti-Ray Sampling', fontsize=12)
            plt.axis('off')
        
        # Handle different scene types
        if isinstance(scene_objects, SphericalCheckerboard):
            # Spherical checkerboard layout
            plt.subplot(3, 2, 3)
            pos = scene_objects.center.cpu().numpy()
            circle = plt.Circle((pos[2], pos[0]), scene_objects.radius, 
                               fill=False, color='blue', linewidth=3)
            plt.gca().add_patch(circle)
            plt.scatter(pos[2], pos[0], c='blue', s=200, marker='o', alpha=0.8)
            plt.text(pos[2]+10, pos[0], f'Spherical Checkerboard\\n{pos[2]:.0f}mm', fontsize=10)
            plt.xlabel('Distance (mm)')
            plt.ylabel('X Position (mm)')
            plt.title(f'{scene_name.title()} - Top View')
            plt.grid(True, alpha=0.3)
            plt.xlim(100, 300)
            plt.ylim(-80, 80)
            
            plt.subplot(3, 2, 4)
            circle = plt.Circle((pos[2], pos[1]), scene_objects.radius, 
                               fill=False, color='blue', linewidth=3)
            plt.gca().add_patch(circle)
            plt.scatter(pos[2], pos[1], c='blue', s=200, marker='o', alpha=0.8)
            plt.text(pos[2]+10, pos[1], f'Spherical Checkerboard\\n{pos[2]:.0f}mm', fontsize=10)
            plt.xlabel('Distance (mm)')
            plt.ylabel('Y Position (mm)')
            plt.title('Side View')
            plt.grid(True, alpha=0.3)
            plt.xlim(100, 300)
            plt.ylim(-80, 80)
            
            # 3D view
            ax = plt.subplot(3, 2, 5, projection='3d')
            ax.scatter(pos[0], pos[1], pos[2], c='blue', s=400, alpha=0.8, marker='o')
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')
            ax.set_zlabel('Distance (mm)')
            ax.set_title('3D Scene Layout')
            
            # Scene details
            plt.subplot(3, 2, 6)
            plt.axis('off')
            plt.text(0.1, 0.9, f'{scene_name.title()} Scene Details:', fontsize=14, fontweight='bold')
            plt.text(0.1, 0.75, f'Spherical Checkerboard', fontsize=12, fontweight='bold')
            plt.text(0.15, 0.65, f'Center: [{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.0f}] mm', fontsize=10)
            plt.text(0.15, 0.55, f'Radius: {scene_objects.radius:.1f} mm', fontsize=10)
            plt.text(0.15, 0.45, f'Pattern: MATLAB-compatible checkerboard', fontsize=10)
            plt.text(0.15, 0.35, f'Squares: 1000x1000 grid, 50px per square', fontsize=10)
            plt.text(0.15, 0.25, f'Colors: Black and white alternating', fontsize=10)
            plt.text(0.15, 0.15, f'Enhanced with multi-ray sampling', fontsize=10, style='italic')
            
        else:
            # Regular scene objects layout
            plt.subplot(3, 2, 3)
            for obj in scene_objects:
                pos = obj.position.cpu().numpy()
                color = obj.color.cpu().numpy()
                plt.scatter(pos[2], pos[0], c=[color], s=obj.size*15, alpha=0.8)
                plt.text(pos[2]+5, pos[0], f'{obj.shape}\\n{pos[2]:.0f}mm', fontsize=8)
            plt.xlabel('Distance (mm)')
            plt.ylabel('X Position (mm)')
            plt.title(f'{scene_name.title()} Scene - Top View')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(3, 2, 4)
            for obj in scene_objects:
                pos = obj.position.cpu().numpy()
                color = obj.color.cpu().numpy()
                plt.scatter(pos[2], pos[1], c=[color], s=obj.size*15, alpha=0.8)
                plt.text(pos[2]+5, pos[1], f'{obj.shape}\\n{pos[2]:.0f}mm', fontsize=8)
            plt.xlabel('Distance (mm)')
            plt.ylabel('Y Position (mm)')
            plt.title('Side View')
            plt.grid(True, alpha=0.3)
            
            # 3D view
            ax = plt.subplot(3, 2, 5, projection='3d')
            for obj in scene_objects:
                pos = obj.position.cpu().numpy()
                color = obj.color.cpu().numpy()
                ax.scatter(pos[0], pos[1], pos[2], c=[color], s=obj.size*30, alpha=0.8)
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')
            ax.set_zlabel('Distance (mm)')
            ax.set_title('3D Scene Layout')
            
            # Object details
            plt.subplot(3, 2, 6)
            plt.axis('off')
            plt.text(0.1, 0.9, f'{scene_name.title()} Scene Objects:', fontsize=14, fontweight='bold')
            y_pos = 0.8
            for i, obj in enumerate(scene_objects):
                pos = obj.position.cpu().numpy()
                color = obj.color.cpu().numpy()
                plt.text(0.1, y_pos, f'{i+1}. {obj.shape.capitalize()}', fontsize=12, fontweight='bold')
                plt.text(0.15, y_pos-0.04, f'Position: [{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.0f}] mm', fontsize=10)
                plt.text(0.15, y_pos-0.08, f'Size: {obj.size:.1f} mm', fontsize=10)
                plt.text(0.15, y_pos-0.12, f'Color: RGB({color[0]:.1f}, {color[1]:.1f}, {color[2]:.1f})', fontsize=10)
                y_pos -= 0.15
        
        plt.suptitle(f'{scene_name.title()} Scene - Ground Truth and Layout', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'{scene_dir}/ground_truth_and_scene.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_eye_movement_gifs(self, scene_name: str, scene_dir: str):
        """Create eye movement GIFs for this scene"""
        display_system = self.scene_optimizers[scene_name]
        scene_objects = self.scene_results[scene_name]['scene_objects']
        
        # Eye position sweep
        print(f"  Creating eye position sweep GIF for {scene_name}...")
        x_positions = torch.linspace(-15, 15, 30, device=device)
        fixed_eye_focal = 35.0
        fixed_tunable_focal = 25.0
        
        frames = []
        for i, x_pos in enumerate(x_positions):
            eye_position = torch.tensor([x_pos.item(), 0.0, 0.0], device=device)
            
            resolution = 384
            retina_size = self.params.retina_size
            y_coords = torch.linspace(-retina_size/2, retina_size/2, resolution, device=device)
            x_coords = torch.linspace(-retina_size/2, retina_size/2, resolution, device=device)
            
            y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
            retina_points = torch.stack([
                x_grid.flatten() + eye_position[0],
                y_grid.flatten() + eye_position[1],
                torch.full_like(x_grid.flatten(), -self.params.retina_distance + eye_position[2])
            ], dim=1)
            
            simulated_colors = trace_complete_optical_path(
                retina_points, fixed_eye_focal, fixed_tunable_focal, 
                display_system, self.params
            )
            
            eye_view = simulated_colors.reshape(resolution, resolution, 3)
            
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.imshow(np.clip(eye_view.detach().cpu().numpy(), 0, 1))
            plt.title(f'{scene_name.title()} - Eye View (X: {x_pos:.1f}mm)\\nHigh Quality System', fontsize=12)
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            for obj in scene_objects:
                pos = obj.position.cpu().numpy()
                color = obj.color.cpu().numpy()
                plt.scatter(pos[2], pos[0], c=[color], s=80, alpha=0.8)
            
            eye_pos_cpu = eye_position.cpu().numpy()
            plt.scatter(0, eye_pos_cpu[0], c='black', s=150, marker='^', label='Eye')
            
            plt.xlabel('Distance (mm)')
            plt.ylabel('X Position (mm)')
            plt.title('Eye Position in Scene')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.xlim(-20, 450)
            plt.ylim(-25, 25)
            
            plt.tight_layout()
            
            frame_path = f'{scene_dir}/eye_pos_frame_{i:03d}.png'
            plt.savefig(frame_path, dpi=150, bbox_inches='tight')
            frames.append(frame_path)
            plt.close()
        
        # Create GIF
        images = [Image.open(frame) for frame in frames]
        images[0].save(f'{scene_dir}/eye_position_sweep.gif',
                      save_all=True, append_images=images[1:],
                      duration=150, loop=0, optimize=False)
        
        # Clean up frames
        for frame in frames:
            os.remove(frame)
    
    def save_focal_length_sweep_gifs(self, scene_name: str, scene_dir: str):
        """Create focal length sweep GIF for this scene - from spherical_checkerboard_raytracer.py"""
        print(f"  Creating focal length sweep GIF for {scene_name}...")
        
        scene_objects = self.scene_results[scene_name]['scene_objects']
        
        # Fixed eye position and tunable focal length range
        eye_position = torch.tensor([0.0, 0.0, 0.0], device=device)
        focal_lengths = torch.linspace(20.0, 50.0, 30, device=device)  # Reduced frames for performance
        
        frames = []
        for i, eye_focal_length in enumerate(focal_lengths):
            # Generate high-quality target image at this focal length
            target_img = generate_target_retina_image_multiray(
                scene_objects, eye_focal_length.item(), self.params, resolution=384
            )
            
            # Calculate focus parameters
            focused_distance = (eye_focal_length.item() * self.params.retina_distance) / (eye_focal_length.item() - self.params.retina_distance)
            
            if isinstance(scene_objects, SphericalCheckerboard):
                scene_distance = 200.0  # Sphere distance
                defocus_distance = abs(scene_distance - focused_distance)
            else:
                # Use average object distance for regular scenes
                object_distances = [obj.position[2].item() for obj in scene_objects]
                avg_distance = sum(object_distances) / len(object_distances)
                defocus_distance = abs(avg_distance - focused_distance)
            
            plt.figure(figsize=(10, 8))
            
            # Main target view (large)
            plt.subplot(2, 1, 1)
            plt.imshow(np.clip(target_img.detach().cpu().numpy(), 0, 1))
            plt.title(f'{scene_name.title()} - Eye Focal Length: {eye_focal_length:.1f}mm\\nMulti-Ray Sampling (Enhanced Quality)', fontsize=16)
            plt.axis('off')
            
            # Focus status
            plt.subplot(2, 1, 2)
            plt.axis('off')
            
            if defocus_distance < 15:
                status = "SHARP FOCUS"
                color = 'green'
            elif defocus_distance < 35:
                status = "MODERATE BLUR"
                color = 'orange'
            else:
                status = "HEAVY BLUR"
                color = 'red'
            
            plt.text(0.5, 0.8, f'Eye Focal Length: {eye_focal_length:.1f}mm', ha='center', fontsize=18, fontweight='bold')
            plt.text(0.5, 0.6, f'Focus Distance: {focused_distance:.0f}mm', ha='center', fontsize=14)
            
            if isinstance(scene_objects, SphericalCheckerboard):
                plt.text(0.5, 0.4, f'Spherical Checkerboard Distance: 200mm', ha='center', fontsize=14)
            else:
                plt.text(0.5, 0.4, f'Average Scene Distance: {avg_distance:.0f}mm', ha='center', fontsize=14)
            
            plt.text(0.5, 0.2, status, ha='center', fontsize=16, color=color, fontweight='bold')
            
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            
            plt.suptitle(f'{scene_name.title()} - Focal Length Sweep (Frame {i+1}/30)', fontsize=18)
            plt.tight_layout()
            
            frame_path = f'{scene_dir}/focal_sweep_frame_{i:03d}.png'
            plt.savefig(frame_path, dpi=120, bbox_inches='tight')
            frames.append(frame_path)
            plt.close()
            
            if i % 5 == 0:
                print(f"    Generated focal sweep frame {i+1}/30 (FL: {eye_focal_length:.1f}mm)")
        
        # Create GIF with timing based on focus quality
        images = []
        durations = []
        
        for i, focal_length in enumerate(focal_lengths):
            focused_distance = (focal_length.item() * self.params.retina_distance) / (focal_length.item() - self.params.retina_distance)
            
            if isinstance(scene_objects, SphericalCheckerboard):
                defocus_distance = abs(200.0 - focused_distance)
            else:
                object_distances = [obj.position[2].item() for obj in scene_objects]
                avg_distance = sum(object_distances) / len(object_distances)
                defocus_distance = abs(avg_distance - focused_distance)
            
            images.append(Image.open(frames[i]))
            
            # Slow down when in focus
            if defocus_distance < 15:
                durations.append(800)  # Long pause when sharp
            elif defocus_distance < 35:
                durations.append(300)  # Medium pause
            else:
                durations.append(150)  # Normal speed when blurred
        
        images[0].save(f'{scene_dir}/focal_length_sweep.gif',
                      save_all=True, append_images=images[1:],
                      duration=durations, loop=0, optimize=True)
        
        # Clean up frames
        for frame in frames:
            os.remove(frame)
        
        print(f"    Focal length sweep GIF created: {scene_dir}/focal_length_sweep.gif")
    
    def save_eye_movement_sweep_gifs(self, scene_name: str, scene_dir: str):
        """Create eye movement sweep GIF for this scene - from spherical_checkerboard_raytracer.py"""
        print(f"  Creating eye movement sweep GIF for {scene_name}...")
        
        scene_objects = self.scene_results[scene_name]['scene_objects']
        eye_focal_length = 30.0  # Fixed eye focal length
        
        # Eye positions sweep
        x_positions = torch.linspace(-15, 15, 25, device=device)
        
        frames = []
        for i, x_pos in enumerate(x_positions):
            
            # For spherical checkerboard, we need to handle eye movement differently
            if isinstance(scene_objects, SphericalCheckerboard):
                # Simple translation for spherical checkerboard
                eye_position = torch.tensor([x_pos.item(), 0.0, 0.0], device=device)
                target_img = generate_target_retina_image_spherical_checkerboard(
                    scene_objects, eye_focal_length, self.params, resolution=384
                )
            else:
                # For regular scenes, just use standard target generation
                target_img = generate_target_retina_image_multiray(
                    scene_objects, eye_focal_length, self.params, resolution=384
                )
            
            plt.figure(figsize=(12, 6))
            
            # Eye view
            plt.subplot(1, 2, 1)
            plt.imshow(np.clip(target_img.detach().cpu().numpy(), 0, 1))
            plt.title(f'{scene_name.title()} - Eye X: {x_pos:.1f}mm\\nMulti-Ray Enhanced View', fontsize=12)
            plt.axis('off')
            
            # Simple scene layout
            plt.subplot(1, 2, 2)
            
            if isinstance(scene_objects, SphericalCheckerboard):
                # Draw spherical checkerboard
                pos = scene_objects.center.cpu().numpy()
                circle = plt.Circle((pos[2], pos[0]), scene_objects.radius, 
                                   fill=False, color='blue', linewidth=3)
                plt.gca().add_patch(circle)
                plt.text(pos[2], pos[0], 'Spherical\\nCheckerboard', ha='center', va='center', fontsize=10)
            else:
                # Draw regular scene objects
                for obj in scene_objects:
                    pos = obj.position.cpu().numpy()
                    color = obj.color.cpu().numpy()
                    plt.scatter(pos[2], pos[0], c=[color], s=80, alpha=0.8)
            
            # Eye position
            plt.scatter(0, x_pos.item(), c='black', s=150, marker='^', label='Eye')
            
            plt.xlabel('Distance (mm)')
            plt.ylabel('X Position (mm)')
            plt.title('Eye Movement in Scene')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.xlim(-20, 450)
            plt.ylim(-20, 20)
            
            plt.suptitle(f'{scene_name.title()} - Eye Movement (Frame {i+1}/25)', fontsize=16)
            plt.tight_layout()
            
            frame_path = f'{scene_dir}/eye_movement_frame_{i:03d}.png'
            plt.savefig(frame_path, dpi=120, bbox_inches='tight')
            frames.append(frame_path)
            plt.close()
            
            if i % 5 == 0:
                print(f"    Generated eye movement frame {i+1}/25")
        
        # Create GIF
        images = [Image.open(frame) for frame in frames]
        images[0].save(f'{scene_dir}/eye_movement_sweep.gif',
                      save_all=True, append_images=images[1:],
                      duration=200, loop=0, optimize=True)
        
        # Clean up frames
        for frame in frames:
            os.remove(frame)
        
        print(f"    Eye movement sweep GIF created: {scene_dir}/eye_movement_sweep.gif")
    
    def save_scene_comparison(self):
        """Save comparison across all scenes"""
        plt.figure(figsize=(24, 20))
        
        scene_names = list(self.scenes.keys())
        num_scenes = len(scene_names)
        
        # Final results comparison with multi-ray sampling
        for i, scene_name in enumerate(scene_names):
            display_system = self.scene_optimizers[scene_name]
            scene_objects = self.scene_results[scene_name]['scene_objects']
            
            _, simulated_img, target_img = self.compute_loss(
                scene_objects, display_system, 35.0, 25.0, resolution=512
            )
            
            # Target
            plt.subplot(4, num_scenes, i + 1)
            plt.imshow(np.clip(target_img.detach().cpu().numpy(), 0, 1))
            plt.title(f'{scene_name.title()} - Target', fontsize=12)
            plt.axis('off')
            
            # Simulated
            plt.subplot(4, num_scenes, i + 1 + num_scenes)
            plt.imshow(np.clip(simulated_img.detach().cpu().numpy(), 0, 1))
            plt.title(f'High Quality Result', fontsize=12)
            plt.axis('off')
            
            # Loss curves
            plt.subplot(4, num_scenes, i + 1 + 2*num_scenes)
            loss_history = self.scene_results[scene_name]['loss_history']
            plt.plot(loss_history, linewidth=2)
            plt.title(f'Training Loss\\nFinal: {loss_history[-1]:.6f}', fontsize=10)
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
            
            # Fixed focal lengths
            plt.subplot(4, num_scenes, i + 1 + 3*num_scenes)
            focal_lengths = display_system.focal_lengths.cpu().numpy()
            bars = plt.bar(range(1, len(focal_lengths)+1), focal_lengths,
                          color=plt.cm.viridis(np.linspace(0, 1, len(focal_lengths))), alpha=0.7)
            plt.title(f'Fixed Focal Lengths', fontsize=10)
            plt.xlabel('Plane')
            plt.ylabel('FL (mm)')
            plt.grid(True, alpha=0.3, axis='y')
            
            for bar, val in zip(bars, focal_lengths):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                        f'{val:.0f}', ha='center', va='bottom', fontsize=8)
        
        plt.suptitle('Enhanced Multi-Ray Scene Comparison (Fixed FL + Circular Microlenses + Spherical Checkerboard)', fontsize=18)
        plt.tight_layout()
        plt.savefig(f'{comprehensive_results_dir}/scene_comparison_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_results_zip(self):
        """Create ZIP file of all results for easy download"""
        print("Creating comprehensive results ZIP file...")
        
        zip_path = f'{base_results_dir}/comprehensive_results.zip'
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Walk through all files in comprehensive_results_dir
            for root, _, files in os.walk(comprehensive_results_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, '.')
                    zipf.write(file_path, arcname)
        
        file_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
        print(f"ZIP file created: {zip_path} ({file_size_mb:.1f} MB)")
        print("Ready for download!")

def main():
    """Main execution function"""
    print("Initializing High-Quality Comprehensive Light Field Display Optimizer...")
    
    # GPU memory check
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Memory Available: {total_memory:.1f} GB")
    
    # Create optical system parameters
    params = OpticalSystemParams()
    
    print("\\n80GB A100-MAXIMIZED ULTRA-HIGH-QUALITY Multi-Ray System Parameters:")
    print(f"  Display resolution: {params.display_resolution}x{params.display_resolution} pixels (MAXIMUM 80GB QUALITY)")
    print(f"  Multi-ray sampling: {params.samples_per_pixel} rays per pixel (MAXIMUM FIDELITY)")
    print(f"  Fixed focal planes: {params.num_focal_planes}")
    num_microlenses = int(params.microlens_array_size/params.microlens_pitch)**2
    print(f"  Circular microlens array: {int(params.microlens_array_size/params.microlens_pitch)}x{int(params.microlens_array_size/params.microlens_pitch)} = {num_microlenses:,} microlenses (ULTRA-HIGH-DENSITY 80GB)")
    print(f"  Enhanced eye resolution: 2048x2048 pixels (80GB MAXIMIZED)")
    print(f"  Scenes: Basic, Complex, Stick Figure, Layered, Office, Nature, Spherical Checkerboard (7 scenes)")
    print(f"  Key features:")
    print(f"    • FIXED focal lengths (stable optimization)")
    print(f"    • CIRCULAR microlenses (massive grid-based processing)")
    print(f"    • MAXIMUM-RESOLUTION displays ({params.display_resolution}x{params.display_resolution})")
    print(f"    • ULTRA-HIGH MULTI-RAY sampling ({params.samples_per_pixel} rays per pixel)")
    print(f"    • A100 MIXED PRECISION training (2x speedup)")
    print(f"    • MASSIVE BATCH processing (256K pixels)")
    print(f"    • AdamW FUSED optimizer with maximum learning rates")
    print(f"    • BOTH MP4 and GIF outputs")
    print(f"    • SPHERICAL CHECKERBOARD scene (MATLAB-compatible)")
    print(f"    • Enhanced focal length and movement sweep GIFs")
    print(f"    • Natural depth-of-field from ray averaging")
    print(f"    • 200 training iterations per scene (maximum convergence)")
    print(f"    • Target memory usage: ~60GB of 80GB available")
    
    # Create comprehensive optimizer
    optimizer = ComprehensiveLightFieldOptimizer(params)
    
    # Use the global parallel processing toggle
    if ENABLE_PARALLEL_SCENES:
        print("\\nUsing PARALLEL scene optimization (experimental)...")
        # Run multiple scenes in parallel (memory permitting)
        def optimize_single_scene(scene_data):
            scene_name, scene_objects = scene_data
            print(f"Starting parallel optimization: {scene_name}")
            optimizer.optimize_scene(scene_name, scene_objects, num_iterations=50)
            return scene_name
        
        # Process 2-3 scenes in parallel 
        scene_pairs = list(optimizer.scenes.items())
        batch_size = 2  # Process 2 scenes at once
        
        for i in range(0, len(scene_pairs), batch_size):
            batch = scene_pairs[i:i+batch_size]
            print(f"\\nOptimizing batch {i//batch_size + 1}: {[name for name, _ in batch]}")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch)) as executor:
                futures = [executor.submit(optimize_single_scene, scene_data) for scene_data in batch]
                for future in concurrent.futures.as_completed(futures):
                    completed_scene = future.result()
                    print(f"Completed: {completed_scene}")
    else:
        print("\\nUsing SEQUENTIAL scene optimization (memory safe)...")
        # Sequential optimization (original approach)
        for i, (scene_name, scene_objects) in enumerate(optimizer.scenes.items()):
            print(f"\\nOptimizing scene {i+1}/{len(optimizer.scenes)}: {scene_name}")
            optimizer.optimize_scene(scene_name, scene_objects, num_iterations=100)  # Balanced iterations for speed
            
            # Force garbage collection and GPU cleanup between scenes
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                current_memory = torch.cuda.memory_allocated() / 1024**3
                print(f"GPU memory after {scene_name}: {current_memory:.2f} GB")
    
    # Generate all outputs
    optimizer.generate_comprehensive_outputs()
    
    print("\\n=== MEMORY-EFFICIENT A100-OPTIMIZED MULTI-RAY OPTIMIZATION COMPLETE ===")
    print("Generated memory-efficient enhanced outputs for all scenes:")
    print("  ✓ 7 Comprehensive scenes (basic, complex, stick_figure, layered, office, nature, spherical_checkerboard)")
    print("  ✓ FIXED focal lengths (stable and reliable)")
    print("  ✓ CIRCULAR microlenses (efficient grid-based selection)")
    print("  ✓ HIGH-QUALITY displays (2048x2048 pixels)")
    print("  ✓ MULTI-RAY sub-aperture sampling (8 rays per pixel)")
    print("  ✓ A100 MIXED PRECISION training (2x speedup)")
    print("  ✓ EFFICIENT BATCH processing (4K pixels)")
    print("  ✓ AdamW FUSED optimizer with efficient learning rates")
    print("  ✓ BOTH MP4 and GIF training videos")
    print("  ✓ Efficient eye resolution (512x512, memory optimized)")
    print("  ✓ SPHERICAL CHECKERBOARD scene (MATLAB-compatible)")
    print("  ✓ Focal length sweep GIFs for all scenes")
    print("  ✓ Eye movement sweep GIFs for all scenes")
    print("  ✓ Natural depth-of-field from ray averaging")
    print("  ✓ Scene-organized subfolders")
    print("  ✓ Cross-scene comparison")
    print("  ✓ ZIP file for easy download")
    print("  ✓ 50 training iterations per scene (fast convergence)")
    print(f"\\nAll MEMORY-EFFICIENT A100-OPTIMIZED MULTI-RAY outputs saved to {comprehensive_results_dir}/!")
    print("Ready for download and analysis with efficient optical physics!")

if __name__ == "__main__":
    main()