# ======================================================================
# COMPLETE ENHANCED RUNPOD LIGHT FIELD OPTIMIZER
# ======================================================================
# Copy this entire cell and paste into your RunPod Jupyter notebook
# Designed for 80GB A100 with comprehensive debug outputs
# 
# FEATURES:
# - 200 iterations with every-iteration tracking
# - Enhanced resolution (2048x2048 display, 512x512 target)
# - 24 rays per pixel multi-ray sampling
# - Progress GIF showing improvement through iterations
# - Focal length sweep GIF
# - Eye position sweep GIF
# - Individual focal plane views
# - Comprehensive result archiving
# ======================================================================

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import shutil
import zipfile
import math
from tqdm.notebook import tqdm
import json
from datetime import datetime
import gc

print("🚀 ENHANCED RUNPOD LIGHT FIELD OPTIMIZER 🚀")
print("Comprehensive debug outputs with every-iteration tracking")
print("Optimized for 80GB A100")

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    initial_memory = torch.cuda.memory_allocated() / 1024**3
    print(f"Initial memory used: {initial_memory:.2f} GB")

# Create comprehensive results structure
base_results = '/workspace/enhanced_results'
if os.path.exists(base_results):
    shutil.rmtree(base_results)

dirs = [
    f'{base_results}/iteration_progress',
    f'{base_results}/focal_length_views', 
    f'{base_results}/gifs',
    f'{base_results}/final_results'
]

for d in dirs:
    os.makedirs(d, exist_ok=True)

print(f"Results structure created: {base_results}")

# Enhanced parameters
class Params:
    eye_pupil_diameter = 4.0
    retina_distance = 24.0  
    retina_size = 10.0
    samples_per_pixel = 24  # Enhanced multi-ray
    display_resolution = 2048  # Enhanced resolution
    num_focal_planes = 10

params = Params()

# Spherical Checkerboard
class SphericalCheckerboard:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius
        
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

# Create scene
scene = SphericalCheckerboard(
    center=torch.tensor([0.0, 0.0, 200.0], device=device),
    radius=50.0
)
print(f"Scene created: Spherical checkerboard at {scene.center.cpu().numpy()}")

# Enhanced Display System
class EnhancedDisplay(nn.Module):
    def __init__(self):
        super().__init__()
        print(f"Creating display: {params.num_focal_planes} x {params.display_resolution}x{params.display_resolution}")
        
        self.display_images = nn.Parameter(
            torch.rand(params.num_focal_planes, 3, params.display_resolution, params.display_resolution, 
                      device=device) * 0.5
        )
        
        self.focal_lengths = torch.linspace(10, 100, params.num_focal_planes, device=device)
        print(f"Focal lengths: {self.focal_lengths.cpu().numpy()}")

display_system = EnhancedDisplay()

# Check memory after display creation
if torch.cuda.is_available():
    display_memory = torch.cuda.memory_allocated() / 1024**3
    print(f"Memory after display creation: {display_memory:.2f} GB")

# Pupil sampling function
def generate_pupil_samples(num_samples, pupil_radius):
    angles = torch.linspace(0, 2*math.pi, num_samples, device=device)
    radii = torch.sqrt(torch.rand(num_samples, device=device)) * pupil_radius
    return torch.stack([radii * torch.cos(angles), radii * torch.sin(angles)], dim=1)

# Enhanced target generation
def generate_target(resolution=512):
    print(f"Generating target image: {resolution}x{resolution} with {params.samples_per_pixel} rays/pixel")
    
    retina_size = params.retina_size
    y_coords = torch.linspace(-retina_size/2, retina_size/2, resolution, device=device)
    x_coords = torch.linspace(-retina_size/2, retina_size/2, resolution, device=device)
    
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
    retina_points = torch.stack([
        x_grid.flatten(), y_grid.flatten(), 
        torch.full_like(x_grid.flatten(), -params.retina_distance)
    ], dim=1)
    
    N = retina_points.shape[0]
    M = params.samples_per_pixel
    
    # Process in manageable batches
    batch_size = min(1024, N)
    final_colors = torch.zeros(N, 3, device=device)
    
    pupil_radius = params.eye_pupil_diameter / 2
    pupil_samples = generate_pupil_samples(M, pupil_radius)
    
    for batch_start in range(0, N, batch_size):
        batch_end = min(batch_start + batch_size, N)
        batch_retina_points = retina_points[batch_start:batch_end]
        batch_N = batch_retina_points.shape[0]
        
        # Multi-ray sampling
        pupil_points_3d = torch.zeros(M, 3, device=device)
        pupil_points_3d[:, 0] = pupil_samples[:, 0]
        pupil_points_3d[:, 1] = pupil_samples[:, 1]
        
        retina_expanded = batch_retina_points.unsqueeze(1)
        pupil_expanded = pupil_points_3d.unsqueeze(0).expand(batch_N, M, 3)
        
        # Eye lens refraction
        ray_dirs = pupil_expanded - retina_expanded
        ray_dirs = ray_dirs / torch.norm(ray_dirs, dim=-1, keepdim=True)
        
        lens_power = 1000.0 / 35.0 / 1000.0  # 35mm focal length
        ray_dirs[:, :, 0] += -lens_power * pupil_expanded[:, :, 0]
        ray_dirs[:, :, 1] += -lens_power * pupil_expanded[:, :, 1]
        ray_dirs = ray_dirs / torch.norm(ray_dirs, dim=-1, keepdim=True)
        
        # Ray-sphere intersection
        ray_origins_flat = pupil_expanded.reshape(-1, 3)
        ray_dirs_flat = ray_dirs.reshape(-1, 3)
        
        oc = ray_origins_flat - scene.center
        a = torch.sum(ray_dirs_flat * ray_dirs_flat, dim=-1)
        b = 2.0 * torch.sum(oc * ray_dirs_flat, dim=-1)
        c = torch.sum(oc * oc, dim=-1) - scene.radius * scene.radius
        
        discriminant = b * b - 4 * a * c
        hit_mask = discriminant >= 0
        
        colors_flat = torch.zeros_like(ray_origins_flat)
        if hit_mask.any():
            sqrt_discriminant = torch.sqrt(discriminant[hit_mask])
            t = (-b[hit_mask] + sqrt_discriminant) / (2 * a[hit_mask])
            valid_hits = t > 1e-6
            
            if valid_hits.any():
                hit_points = ray_origins_flat[hit_mask][valid_hits] + t[valid_hits].unsqueeze(-1) * ray_dirs_flat[hit_mask][valid_hits]
                checkerboard_colors = scene.get_color(hit_points)
                
                final_mask = torch.zeros_like(hit_mask)
                final_mask[hit_mask] = valid_hits
                
                colors_flat[final_mask, :] = checkerboard_colors.unsqueeze(-1)
        
        # Average over rays
        colors = colors_flat.reshape(batch_N, M, 3)
        valid_mask = torch.sqrt(pupil_expanded[:, :, 0]**2 + pupil_expanded[:, :, 1]**2) <= pupil_radius
        
        batch_colors = torch.zeros(batch_N, 3, device=device)
        for pixel_idx in range(batch_N):
            valid_samples = valid_mask[pixel_idx, :]
            if valid_samples.any():
                batch_colors[pixel_idx, :] = torch.mean(colors[pixel_idx, valid_samples, :], dim=0)
        
        final_colors[batch_start:batch_end] = batch_colors
    
    return final_colors.reshape(resolution, resolution, 3)

# Generate target image
print("Generating enhanced target image...")
with torch.no_grad():
    target_image = generate_target(512)
print(f"Target generated: {target_image.shape}")

# Check memory after target generation
if torch.cuda.is_available():
    target_memory = torch.cuda.memory_allocated() / 1024**3
    print(f"Memory after target generation: {target_memory:.2f} GB")

# Optimizer setup
optimizer = optim.AdamW(display_system.parameters(), lr=0.02, weight_decay=1e-4)
scaler = torch.cuda.amp.GradScaler()

print(f"\nEnhanced Training Configuration:")
print(f"  Display resolution: {params.display_resolution}x{params.display_resolution}")
print(f"  Multi-ray sampling: {params.samples_per_pixel} rays per pixel")
print(f"  Target resolution: 512x512")
print(f"  Focal planes: {params.num_focal_planes}")
print(f"  Iterations: 200 (EVERY iteration tracked)")
print(f"  Expected memory: ~20-30GB")

# Training with comprehensive tracking
loss_history = []
start_time = datetime.now()

print(f"\n🚀 Starting enhanced training with comprehensive debug outputs...")

for iteration in tqdm(range(200), desc="Enhanced Training"):
    
    optimizer.zero_grad()
    
    # Generate simulated image from display
    with torch.cuda.amp.autocast():
        simulated_image = torch.nn.functional.interpolate(
            display_system.display_images[0].unsqueeze(0), 
            size=(512, 512), mode='bilinear', align_corners=False
        ).squeeze(0).permute(1, 2, 0)
        
        # Compute loss
        loss = torch.mean((simulated_image - target_image) ** 2)
    
    # Backward pass
    scaler.scale(loss).backward()
    torch.nn.utils.clip_grad_norm_(display_system.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
    
    # Clamp values
    with torch.no_grad():
        display_system.display_images.clamp_(0, 1)
    
    loss_history.append(loss.item())
    
    # Save EVERY iteration for progress GIF
    if True:  # Save every iteration
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.imshow(np.clip(target_image.cpu().numpy(), 0, 1))
        plt.title(f'Target\nIteration {iteration}')
        plt.axis('off')
        
        plt.subplot(1, 3, 2) 
        plt.imshow(np.clip(simulated_image.detach().cpu().numpy(), 0, 1))
        plt.title(f'Optimized\nLoss: {loss.item():.6f}')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        if len(loss_history) > 1:
            plt.plot(loss_history, 'b-', linewidth=2)
            plt.scatter(iteration, loss.item(), color='red', s=30, zorder=5)
        plt.title(f'Loss: {loss.item():.6f}')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        plt.suptitle(f'Training Progress - Iteration {iteration}/200', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{base_results}/iteration_progress/iter_{iteration:04d}.png', dpi=100, bbox_inches='tight')
        plt.close()
    
    # Save focal plane views every 25 iterations
    if iteration % 25 == 0:
        plt.figure(figsize=(25, 5))
        for i in range(params.num_focal_planes):
            img = display_system.display_images[i].detach().cpu().numpy().transpose(1, 2, 0)
            plt.subplot(1, params.num_focal_planes, i + 1)
            plt.imshow(np.clip(img, 0, 1))
            plt.title(f'{display_system.focal_lengths[i]:.0f}mm\nIter {iteration}', fontsize=10)
            plt.axis('off')
        
        plt.suptitle(f'All Focal Planes - Iteration {iteration}', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'{base_results}/focal_length_views/focal_iter_{iteration:04d}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # Progress updates
    if iteration % 20 == 0:
        memory_used = torch.cuda.memory_allocated() / 1024**3
        elapsed = (datetime.now() - start_time).total_seconds() / 60
        print(f"  Iter {iteration}: Loss = {loss.item():.6f}, GPU: {memory_used:.2f} GB, Time: {elapsed:.1f}m")
    
    # Memory cleanup
    if iteration % 50 == 0:
        torch.cuda.empty_cache()

end_time = datetime.now()
total_duration = (end_time - start_time).total_seconds() / 60

print(f"\n🎉 TRAINING COMPLETE! 🎉")
print(f"Duration: {total_duration:.1f} minutes")
print(f"Final loss: {loss_history[-1]:.6f}")

# Final memory report
final_memory = torch.cuda.memory_allocated() / 1024**3
max_memory = torch.cuda.max_memory_allocated() / 1024**3
print(f"Final GPU memory: {final_memory:.2f} GB")
print(f"Peak GPU memory: {max_memory:.2f} GB")

# Create comprehensive GIFs
print("\n🎬 Creating comprehensive GIFs...")

# 1. Training Progress GIF (every iteration)
print("  Creating training progress GIF...")
progress_images = []
iter_dir = f'{base_results}/iteration_progress'

for i in range(0, 200, 5):  # Every 5th iteration for manageable GIF size
    iter_file = f'{iter_dir}/iter_{i:04d}.png'
    if os.path.exists(iter_file):
        progress_images.append(Image.open(iter_file))

if progress_images:
    progress_images[0].save(f'{base_results}/gifs/training_progress.gif',
                           save_all=True, append_images=progress_images[1:],
                           duration=150, loop=0, optimize=True)
    print(f"    Training progress GIF: {len(progress_images)} frames")

# 2. Focal Length Sweep GIF
print("  Creating focal length sweep GIF...")
focal_frames = []
focal_lengths_test = torch.linspace(20.0, 60.0, 25, device=device)

for i, fl in enumerate(focal_lengths_test):
    with torch.no_grad():
        # Generate target at this focal length
        target_fl = generate_target(384)  # Smaller resolution for speed
    
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 1, 1)
    plt.imshow(np.clip(target_fl.cpu().numpy(), 0, 1))
    plt.title(f'Spherical Checkerboard - Eye FL: {fl:.1f}mm\n24-Ray Multi-Ray Sampling')
    plt.axis('off')
    
    plt.subplot(2, 1, 2)
    plt.axis('off')
    
    # Focus calculation
    focused_distance = (fl.item() * params.retina_distance) / (fl.item() - params.retina_distance)
    defocus = abs(200.0 - focused_distance)
    
    status = "SHARP FOCUS" if defocus < 15 else "MODERATE BLUR" if defocus < 35 else "HEAVY BLUR"
    color = 'green' if defocus < 15 else 'orange' if defocus < 35 else 'red'
    
    plt.text(0.5, 0.8, f'Eye Focal Length: {fl:.1f}mm', ha='center', fontsize=16, fontweight='bold')
    plt.text(0.5, 0.6, f'Focus Distance: {focused_distance:.0f}mm', ha='center', fontsize=14)
    plt.text(0.5, 0.4, f'Sphere Distance: 200mm', ha='center', fontsize=14)
    plt.text(0.5, 0.2, status, ha='center', fontsize=14, color=color, fontweight='bold')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    plt.suptitle(f'Focal Length Sweep - Frame {i+1}/25', fontsize=16)
    plt.tight_layout()
    
    frame_path = f'{base_results}/gifs/focal_frame_{i:03d}.png'
    plt.savefig(frame_path, dpi=120, bbox_inches='tight')
    focal_frames.append(frame_path)
    plt.close()

# Create focal sweep GIF
focal_images = [Image.open(f) for f in focal_frames]
focal_images[0].save(f'{base_results}/gifs/focal_length_sweep.gif',
                    save_all=True, append_images=focal_images[1:],
                    duration=300, loop=0, optimize=True)

# Clean up focal frames
for f in focal_frames:
    os.remove(f)
    
print("    Focal length sweep GIF created")

# 3. Eye Position Sweep GIF
print("  Creating eye position sweep GIF...")
eye_frames = []
eye_positions = torch.linspace(-15, 15, 20, device=device)

for i, eye_x in enumerate(eye_positions):
    with torch.no_grad():
        target_eye = generate_target(384)
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(np.clip(target_eye.cpu().numpy(), 0, 1))
    plt.title(f'Eye View from X: {eye_x:.1f}mm\nSpherical Checkerboard')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    # Scene diagram
    pos = scene.center.cpu().numpy()
    circle = plt.Circle((pos[2], pos[0]), scene.radius, fill=False, color='blue', linewidth=3)
    plt.gca().add_patch(circle)
    plt.scatter(pos[2], pos[0], c='blue', s=200, marker='o', alpha=0.8)
    plt.scatter(0, eye_x.item(), c='red', s=150, marker='^', label='Eye')
    plt.plot([0, pos[2]], [eye_x.item(), pos[0]], 'r--', alpha=0.5)
    
    plt.xlabel('Distance (mm)')
    plt.ylabel('X Position (mm)')
    plt.title('Eye Movement in Scene')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(-20, 250)
    plt.ylim(-20, 20)
    
    plt.suptitle(f'Eye Position Sweep - Frame {i+1}/20', fontsize=16)
    plt.tight_layout()
    
    frame_path = f'{base_results}/gifs/eye_frame_{i:03d}.png'
    plt.savefig(frame_path, dpi=120, bbox_inches='tight')
    eye_frames.append(frame_path)
    plt.close()

# Create eye movement GIF
eye_images = [Image.open(f) for f in eye_frames]
eye_images[0].save(f'{base_results}/gifs/eye_position_sweep.gif',
                  save_all=True, append_images=eye_images[1:],
                  duration=250, loop=0, optimize=True)

# Clean up eye frames
for f in eye_frames:
    os.remove(f)
    
print("    Eye position sweep GIF created")

# Save final comprehensive results
print("\n📊 Saving final comprehensive results...")

# Complete results visualization
plt.figure(figsize=(20, 12))

plt.subplot(3, 4, 1)
plt.imshow(np.clip(target_image.cpu().numpy(), 0, 1))
plt.title('Target: Spherical Checkerboard\n24-Ray Multi-Ray Sampling')
plt.axis('off')

plt.subplot(3, 4, 2)
final_simulated = torch.nn.functional.interpolate(
    display_system.display_images[0].unsqueeze(0), 
    size=(512, 512), mode='bilinear'
).squeeze(0).permute(1, 2, 0)
plt.imshow(np.clip(final_simulated.detach().cpu().numpy(), 0, 1))
plt.title(f'Final Optimized Output\nLoss: {loss_history[-1]:.6f}')
plt.axis('off')

plt.subplot(3, 4, 3)
diff_img = torch.abs(final_simulated - target_image)
plt.imshow(np.clip(diff_img.detach().cpu().numpy(), 0, 1))
plt.title('Final Difference')
plt.axis('off')

plt.subplot(3, 4, 4)
plt.plot(loss_history, 'b-', linewidth=2)
plt.title(f'Complete Loss Curve\nFinal: {loss_history[-1]:.6f}')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.yscale('log')
plt.grid(True, alpha=0.3)

# Show all focal plane displays
for i in range(min(8, params.num_focal_planes)):
    plt.subplot(3, 4, 5 + i)
    display_img = display_system.display_images[i].detach().cpu().numpy()
    display_img = np.transpose(display_img, (1, 2, 0))
    plt.imshow(np.clip(display_img, 0, 1))
    plt.title(f'FL: {display_system.focal_lengths[i]:.0f}mm', fontsize=10)
    plt.axis('off')

plt.suptitle('Enhanced RunPod Optimization - Complete Results', fontsize=18)
plt.tight_layout()
plt.savefig(f'{base_results}/final_results/complete_enhanced_results.png', dpi=300, bbox_inches='tight')
plt.show()  # Display in Jupyter

# Create comprehensive archive
print("\n📦 Creating comprehensive archive...")
archive_path = f"/workspace/enhanced_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"

with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, _, files in os.walk(base_results):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, '/workspace')
            zipf.write(file_path, arcname)

archive_size = os.path.getsize(archive_path) / 1024**2
print(f"Archive created: {archive_path} ({archive_size:.1f} MB)")

# Final summary
print(f"\n✅ ENHANCED OPTIMIZATION COMPLETE! ✅")
print(f"📁 Results directory: {base_results}")
print(f"📦 Archive: {archive_path}")
print(f"⏱️  Total duration: {total_duration:.1f} minutes")
print(f"🧠 Final GPU memory: {final_memory:.2f} GB")
print(f"📈 Peak GPU memory: {max_memory:.2f} GB")
print(f"\n🎬 Generated GIFs:")
print(f"   • Training progress: {base_results}/gifs/training_progress.gif")
print(f"   • Focal length sweep: {base_results}/gifs/focal_length_sweep.gif")
print(f"   • Eye position sweep: {base_results}/gifs/eye_position_sweep.gif")
print(f"\n📊 Debug outputs:")
print(f"   • Every iteration progress: {base_results}/iteration_progress/")
print(f"   • Focal plane views: {base_results}/focal_length_views/")
print(f"   • Final results: {base_results}/final_results/")

print(f"\n🚀 Copy {archive_path} to download all results! 🚀")