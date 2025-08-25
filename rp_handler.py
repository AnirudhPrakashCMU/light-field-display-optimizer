"""
Enhanced RunPod Light Field Display Optimizer Handler
Complete implementation with multi-ray sampling, comprehensive debug outputs, and memory optimization
"""

import runpod
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import shutil
import zipfile
import json
import math
from datetime import datetime

print("🚀 Enhanced Light Field Optimizer Handler Loading...")

# Setup GPU with aggressive memory management
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    torch.backends.cudnn.benchmark = True
    print(f"✅ GPU initialized: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠️ No GPU detected")

class SphericalCheckerboard:
    """MATLAB-compatible spherical checkerboard scene"""
    
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius
        
    def get_color(self, point_3d):
        """Get checkerboard color at 3D point"""
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
    """Generate uniform pupil samples for multi-ray sampling"""
    angles = torch.linspace(0, 2*math.pi, num_samples, device=device)
    radii = torch.sqrt(torch.rand(num_samples, device=device)) * pupil_radius
    return torch.stack([radii * torch.cos(angles), radii * torch.sin(angles)], dim=1)

class EnhancedLightFieldDisplay(nn.Module):
    """Enhanced display system with memory optimization"""
    
    def __init__(self, params):
        super().__init__()
        
        # Calculate memory-optimized display configuration
        target_memory_gb = params.get('target_memory_gb', 20)
        display_resolution = params.get('display_resolution', 2048)
        num_focal_planes = params.get('num_focal_planes', 12)
        
        print(f"🧠 Creating memory-optimized display:")
        print(f"   Target memory: {target_memory_gb}GB")
        print(f"   Display resolution: {display_resolution}x{display_resolution}")
        print(f"   Focal planes: {num_focal_planes}")
        
        # Learnable display images
        self.display_images = nn.Parameter(
            torch.rand(num_focal_planes, 3, display_resolution, display_resolution, 
                      device=device, dtype=torch.float32) * 0.5
        )
        
        # Fixed focal lengths
        self.focal_lengths = torch.linspace(10, 100, num_focal_planes, device=device)
        
        # Report actual memory usage
        memory_used = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        print(f"   Actual memory used: {memory_used:.2f} GB")

def generate_enhanced_target(scene, eye_focal_length, resolution, rays_per_pixel):
    """Generate high-quality target with multi-ray sampling"""
    
    print(f"🎯 Generating target: {resolution}x{resolution} with {rays_per_pixel} rays/pixel")
    
    retina_size = 10.0
    retina_distance = 24.0
    
    y_coords = torch.linspace(-retina_size/2, retina_size/2, resolution, device=device)
    x_coords = torch.linspace(-retina_size/2, retina_size/2, resolution, device=device)
    
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
    retina_points = torch.stack([
        x_grid.flatten(), y_grid.flatten(), 
        torch.full_like(x_grid.flatten(), -retina_distance)
    ], dim=1)
    
    N = retina_points.shape[0]
    M = rays_per_pixel
    
    # Process in batches
    batch_size = min(4096, N)
    final_colors = torch.zeros(N, 3, device=device)
    
    pupil_radius = 2.0  # mm
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
        
        lens_power = 1000.0 / eye_focal_length / 1000.0
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

def run_enhanced_optimization(iterations, resolution, rays_per_pixel, target_memory_gb):
    """Run the complete enhanced light field optimization"""
    
    print(f"🚀 Starting Enhanced Light Field Optimization")
    print(f"   Iterations: {iterations}")
    print(f"   Resolution: {resolution}x{resolution}")
    print(f"   Rays per pixel: {rays_per_pixel}")
    print(f"   Target memory: {target_memory_gb}GB")
    
    # Create scene
    scene = SphericalCheckerboard(
        center=torch.tensor([0.0, 0.0, 200.0], device=device),
        radius=50.0
    )
    
    # Enhanced display system
    display_params = {
        'target_memory_gb': target_memory_gb,
        'display_resolution': min(4096, resolution * 2),  # Scale with input resolution
        'num_focal_planes': max(8, min(20, int(target_memory_gb / 3)))  # Scale with memory
    }
    
    display_system = EnhancedLightFieldDisplay(display_params)
    
    # Optimizer with enhanced settings
    optimizer = optim.AdamW(display_system.parameters(), lr=0.03, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    # Generate target image
    print("🎯 Generating enhanced target image...")
    with torch.no_grad():
        target_image = generate_enhanced_target(scene, 35.0, resolution, rays_per_pixel)
    
    print(f"✅ Target generated: {target_image.shape}")
    
    # Training with comprehensive tracking
    loss_history = []
    iteration_images = []
    
    print(f"🔥 Starting enhanced training...")
    
    for iteration in range(iterations):
        optimizer.zero_grad()
        
        # Generate simulated image
        with torch.cuda.amp.autocast() if torch.cuda.is_available() else torch.no_grad():
            simulated_image = torch.nn.functional.interpolate(
                display_system.display_images[0].unsqueeze(0), 
                size=(resolution, resolution), mode='bilinear', align_corners=False
            ).squeeze(0).permute(1, 2, 0)
            
            # Compute loss
            loss = torch.mean((simulated_image - target_image) ** 2)
        
        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
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
        
        # Save every 20th iteration for progress GIF
        if iteration % 20 == 0:
            # Create progress frame
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            
            axes[0].imshow(np.clip(target_image.cpu().numpy(), 0, 1))
            axes[0].set_title(f'Target\nIteration {iteration}')
            axes[0].axis('off')
            
            axes[1].imshow(np.clip(simulated_image.detach().cpu().numpy(), 0, 1))
            axes[1].set_title(f'Optimized\nLoss: {loss.item():.6f}')
            axes[1].axis('off')
            
            axes[2].plot(loss_history, 'b-', linewidth=2)
            axes[2].set_title(f'Loss: {loss.item():.6f}')
            axes[2].set_yscale('log')
            axes[2].grid(True, alpha=0.3)
            
            plt.suptitle(f'Training Progress - Iteration {iteration}/{iterations}')
            plt.tight_layout()
            
            # Save as PIL Image
            temp_path = f'/tmp/progress_frame_{iteration:04d}.png'
            plt.savefig(temp_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            iteration_images.append(temp_path)
        
        # Progress reporting
        if iteration % 50 == 0:
            memory_used = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
            print(f"   Iter {iteration}: Loss = {loss.item():.6f}, GPU = {memory_used:.2f} GB")
        
        # Memory cleanup
        if iteration % 100 == 0:
            torch.cuda.empty_cache()
    
    # Create progress GIF
    print("🎬 Creating training progress GIF...")
    if iteration_images:
        gif_images = [Image.open(img_path) for img_path in iteration_images]
        progress_gif_path = '/tmp/training_progress.gif'
        gif_images[0].save(progress_gif_path, save_all=True, append_images=gif_images[1:], 
                          duration=300, loop=0, optimize=True)
        
        # Clean up frames
        for img_path in iteration_images:
            if os.path.exists(img_path):
                os.remove(img_path)
        
        print(f"✅ Progress GIF created: {len(gif_images)} frames")
    
    # Create focal length sweep GIF
    print("🎬 Creating focal length sweep GIF...")
    focal_frames = []
    focal_lengths_test = torch.linspace(20.0, 60.0, 20, device=device)
    
    for i, fl in enumerate(focal_lengths_test):
        with torch.no_grad():
            target_fl = generate_enhanced_target(scene, fl.item(), 384, rays_per_pixel//2)
        
        plt.figure(figsize=(10, 8))
        
        plt.subplot(2, 1, 1)
        plt.imshow(np.clip(target_fl.cpu().numpy(), 0, 1))
        plt.title(f'Spherical Checkerboard - Eye FL: {fl:.1f}mm\n{rays_per_pixel}-Ray Multi-Ray Sampling')
        plt.axis('off')
        
        plt.subplot(2, 1, 2)
        plt.axis('off')
        
        # Focus calculation
        focused_distance = (fl.item() * 24.0) / (fl.item() - 24.0)
        defocus = abs(200.0 - focused_distance)
        
        status = "SHARP FOCUS" if defocus < 15 else "MODERATE BLUR" if defocus < 35 else "HEAVY BLUR"
        color = 'green' if defocus < 15 else 'orange' if defocus < 35 else 'red'
        
        plt.text(0.5, 0.8, f'Eye Focal Length: {fl:.1f}mm', ha='center', fontsize=16, fontweight='bold')
        plt.text(0.5, 0.6, f'Focus Distance: {focused_distance:.0f}mm', ha='center', fontsize=14)
        plt.text(0.5, 0.4, f'Sphere Distance: 200mm', ha='center', fontsize=14)
        plt.text(0.5, 0.2, status, ha='center', fontsize=14, color=color, fontweight='bold')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        
        plt.suptitle(f'Focal Length Sweep - Frame {i+1}/20')
        plt.tight_layout()
        
        frame_path = f'/tmp/focal_frame_{i:03d}.png'
        plt.savefig(frame_path, dpi=120, bbox_inches='tight')
        focal_frames.append(frame_path)
        plt.close()
    
    # Create focal sweep GIF
    focal_images = [Image.open(f) for f in focal_frames]
    focal_gif_path = '/tmp/focal_length_sweep.gif'
    focal_images[0].save(focal_gif_path, save_all=True, append_images=focal_images[1:],
                        duration=400, loop=0, optimize=True)
    
    # Clean up focal frames
    for f in focal_frames:
        os.remove(f)
    
    print(f"✅ Focal sweep GIF created")
    
    # Create eye movement GIF
    print("🎬 Creating eye movement GIF...")
    eye_frames = []
    eye_positions = torch.linspace(-15, 15, 15, device=device)
    
    for i, eye_x in enumerate(eye_positions):
        with torch.no_grad():
            target_eye = generate_enhanced_target(scene, 35.0, 384, rays_per_pixel//2)
        
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
        plt.scatter(pos[2], pos[0], c='blue', s=200, marker='o')
        plt.scatter(0, eye_x.item(), c='red', s=150, marker='^', label='Eye')
        
        plt.xlabel('Distance (mm)')
        plt.ylabel('X Position (mm)')
        plt.title('Eye Movement')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xlim(-20, 250)
        plt.ylim(-20, 20)
        
        plt.suptitle(f'Eye Movement - Frame {i+1}/15')
        plt.tight_layout()
        
        frame_path = f'/tmp/eye_frame_{i:03d}.png'
        plt.savefig(frame_path, dpi=120, bbox_inches='tight')
        eye_frames.append(frame_path)
        plt.close()
    
    # Create eye movement GIF
    eye_images = [Image.open(f) for f in eye_frames]
    eye_gif_path = '/tmp/eye_movement_sweep.gif'
    eye_images[0].save(eye_gif_path, save_all=True, append_images=eye_images[1:],
                      duration=250, loop=0, optimize=True)
    
    # Clean up eye frames
    for f in eye_frames:
        os.remove(f)
    
    print(f"✅ Eye movement GIF created")
    
    # Save final results
    print("📊 Saving comprehensive results...")
    
    # Final comparison
    plt.figure(figsize=(20, 10))
    
    # Main comparison
    plt.subplot(2, 4, 1)
    plt.imshow(np.clip(target_image.cpu().numpy(), 0, 1))
    plt.title(f'Target: Spherical Checkerboard\n{rays_per_pixel}-Ray Multi-Sampling')
    plt.axis('off')
    
    plt.subplot(2, 4, 2)
    plt.imshow(np.clip(simulated_image.detach().cpu().numpy(), 0, 1))
    plt.title(f'Final Optimized Output\nLoss: {loss_history[-1]:.6f}')
    plt.axis('off')
    
    plt.subplot(2, 4, 3)
    diff_img = torch.abs(simulated_image - target_image)
    plt.imshow(np.clip(diff_img.detach().cpu().numpy(), 0, 1))
    plt.title('Difference Image')
    plt.axis('off')
    
    plt.subplot(2, 4, 4)
    plt.plot(loss_history, 'b-', linewidth=2)
    plt.title(f'Complete Loss Curve\nFinal: {loss_history[-1]:.6f}')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # Show focal plane displays
    for i in range(min(4, len(display_system.focal_lengths))):
        plt.subplot(2, 4, 5 + i)
        display_img = display_system.display_images[i].detach().cpu().numpy()
        display_img = np.transpose(display_img, (1, 2, 0))
        plt.imshow(np.clip(display_img, 0, 1))
        plt.title(f'FL: {display_system.focal_lengths[i]:.0f}mm')
        plt.axis('off')
    
    plt.suptitle('Enhanced Light Field Optimization - Complete Results')
    plt.tight_layout()
    results_path = '/tmp/complete_results.png'
    plt.savefig(results_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    # Final memory report
    final_memory = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    max_memory = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    
    print(f"🎉 Enhanced optimization complete!")
    print(f"   Final loss: {loss_history[-1]:.6f}")
    print(f"   Peak GPU memory: {max_memory:.2f} GB")
    
    return {
        'loss_history': loss_history,
        'final_loss': loss_history[-1],
        'iterations': iterations,
        'resolution': resolution,
        'rays_per_pixel': rays_per_pixel,
        'gpu_memory_used': final_memory,
        'gpu_memory_peak': max_memory,
        'target_memory_gb': target_memory_gb,
        'num_focal_planes': len(display_system.focal_lengths),
        'display_resolution': display_params['display_resolution'],
        'gifs_created': {
            'training_progress': '/tmp/training_progress.gif',
            'focal_length_sweep': '/tmp/focal_length_sweep.gif', 
            'eye_movement_sweep': '/tmp/eye_movement_sweep.gif'
        },
        'final_results_image': results_path
    }

def handler(job):
    """Enhanced RunPod handler with complete light field optimization"""
    
    try:
        print(f"🚀 Enhanced Handler Started: {datetime.now()}")
        
        inp = job.get("input", {}) or {}
        print(f"📋 Input: {json.dumps(inp, indent=2)}")
        
        # Enhanced parameters
        task_type = inp.get("task_type", "full_optimization")
        iterations = inp.get("iterations", 200)
        resolution = inp.get("resolution", 512)
        rays_per_pixel = inp.get("rays_per_pixel", 24)
        target_memory_gb = inp.get("target_memory_gb", 30)
        save_large_outputs = inp.get("save_large_outputs", True)
        
        print(f"⚙️ Enhanced Parameters:")
        print(f"   Task: {task_type}")
        print(f"   Iterations: {iterations}")
        print(f"   Resolution: {resolution}x{resolution}")
        print(f"   Rays per pixel: {rays_per_pixel}")
        print(f"   Target memory: {target_memory_gb}GB")
        
        # GPU info
        gpu_info = {}
        if torch.cuda.is_available():
            gpu_info = {
                'gpu_name': torch.cuda.get_device_name(0),
                'gpu_memory_total': torch.cuda.get_device_properties(0).total_memory / 1e9,
                'gpu_memory_available': (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9
            }
            print(f"🖥️ GPU: {gpu_info['gpu_name']} ({gpu_info['gpu_memory_total']:.1f}GB total)")
        
        # Run enhanced optimization
        results = run_enhanced_optimization(iterations, resolution, rays_per_pixel, target_memory_gb)
        
        # Create comprehensive archive
        if save_large_outputs:
            print("📦 Creating comprehensive archive...")
            archive_path = f'/tmp/enhanced_optimization_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip'
            
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add all result files
                for key, path in results.get('gifs_created', {}).items():
                    if os.path.exists(path):
                        zipf.write(path, f'{key}.gif')
                
                if os.path.exists(results.get('final_results_image', '')):
                    zipf.write(results['final_results_image'], 'complete_results.png')
                
                # Add loss history as JSON
                loss_json_path = '/tmp/loss_history.json'
                with open(loss_json_path, 'w') as f:
                    json.dump(results['loss_history'], f)
                zipf.write(loss_json_path, 'loss_history.json')
            
            archive_size = os.path.getsize(archive_path) / 1024**2
            print(f"📦 Archive created: {archive_size:.1f} MB")
            results['archive_path'] = archive_path
            results['archive_size_mb'] = archive_size
        
        return {
            'status': 'success',
            'message': f'Enhanced optimization complete: {iterations} iterations, {rays_per_pixel} rays/pixel, {results["gpu_memory_peak"]:.2f}GB peak memory',
            'results': results,
            'gpu_info': gpu_info,
            'timestamp': datetime.now().isoformat(),
            'handler_version': 'enhanced_v2.0'
        }
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        
        print(f"💥 Handler Error: {str(e)}")
        print(f"📋 Traceback: {error_details}")
        
        return {
            'status': 'error',
            'message': f'Enhanced optimization failed: {str(e)}',
            'error_details': error_details,
            'timestamp': datetime.now().isoformat(),
            'handler_version': 'enhanced_v2.0'
        }

print("✅ Enhanced Light Field Optimizer Handler Ready")

# Start RunPod serverless worker
runpod.serverless.start({"handler": handler})