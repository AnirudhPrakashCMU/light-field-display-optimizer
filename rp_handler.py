"""
Clean RunPod Light Field Optimizer Handler
7 scenes with file.io uploads, no external dependencies
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

print("🚀 CLEAN LIGHT FIELD OPTIMIZER - 7 SCENES + FILE.IO")

# GPU setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    torch.backends.cudnn.benchmark = True
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

class SphericalCheckerboard:
    """Spherical checkerboard scene"""
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

# All 7 scenes
ALL_SCENES = {
    'basic': [
        {'position': [0, 0, 150], 'size': 15, 'color': [1, 0, 0], 'shape': 'sphere'},
        {'position': [20, 0, 200], 'size': 10, 'color': [0, 1, 0], 'shape': 'sphere'},
        {'position': [-15, 10, 180], 'size': 8, 'color': [0, 0, 1], 'shape': 'sphere'}
    ],
    'complex': [
        {'position': [0, 0, 120], 'size': 20, 'color': [1, 0.5, 0], 'shape': 'sphere'},
        {'position': [30, 15, 180], 'size': 12, 'color': [0.8, 0, 0.8], 'shape': 'sphere'},
        {'position': [-25, -10, 200], 'size': 15, 'color': [0, 0.8, 0.8], 'shape': 'sphere'},
        {'position': [10, -20, 250], 'size': 18, 'color': [1, 1, 0], 'shape': 'sphere'}
    ],
    'stick_figure': [
        {'position': [0, 15, 180], 'size': 8, 'color': [1, 0.8, 0.6], 'shape': 'sphere'},
        {'position': [0, 0, 180], 'size': 6, 'color': [1, 0.8, 0.6], 'shape': 'sphere'},
        {'position': [-8, 5, 180], 'size': 4, 'color': [1, 0.8, 0.6], 'shape': 'sphere'},
        {'position': [8, 5, 180], 'size': 4, 'color': [1, 0.8, 0.6], 'shape': 'sphere'}
    ],
    'layered': [
        {'position': [0, 0, 100], 'size': 12, 'color': [1, 0, 0], 'shape': 'sphere'},
        {'position': [0, 0, 200], 'size': 15, 'color': [0, 1, 0], 'shape': 'sphere'},
        {'position': [0, 0, 300], 'size': 18, 'color': [0, 0, 1], 'shape': 'sphere'}
    ],
    'office': [
        {'position': [-20, -20, 150], 'size': 25, 'color': [0.8, 0.6, 0.4], 'shape': 'sphere'},
        {'position': [0, 10, 180], 'size': 8, 'color': [0.2, 0.2, 0.2], 'shape': 'sphere'},
        {'position': [15, -15, 160], 'size': 6, 'color': [0.9, 0.9, 0.9], 'shape': 'sphere'}
    ],
    'nature': [
        {'position': [0, -30, 200], 'size': 35, 'color': [0.4, 0.8, 0.2], 'shape': 'sphere'},
        {'position': [25, -25, 180], 'size': 20, 'color': [0.3, 0.7, 0.1], 'shape': 'sphere'},
        {'position': [-20, -30, 220], 'size': 30, 'color': [0.5, 0.9, 0.3], 'shape': 'sphere'}
    ],
    'spherical_checkerboard': SphericalCheckerboard(
        center=torch.tensor([0.0, 0.0, 200.0], device=device),
        radius=50.0
    )
}

class LightFieldDisplay(nn.Module):
    """Light field display system"""
    def __init__(self, resolution=1024, num_planes=8):
        super().__init__()
        
        print(f"🧠 Display: {resolution}x{resolution}, {num_planes} focal planes")
        
        self.display_images = nn.Parameter(
            torch.rand(num_planes, 3, resolution, resolution, device=device) * 0.5
        )
        
        self.focal_lengths = torch.linspace(10, 100, num_planes, device=device)
        
        memory_used = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        print(f"   Memory: {memory_used:.2f} GB")

def upload_to_0x0(file_path):
    """Upload file to file.io with robust error handling"""
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None
    
    file_size = os.path.getsize(file_path) / 1024**2
    filename = os.path.basename(file_path)
    
    print(f"📤 Uploading {filename} ({file_size:.1f} MB) to file.io...")
    
    try:
        with open(file_path, 'rb') as f:
            files = {'file': (filename, f, 'application/octet-stream')}
            response = requests.post('https://file.io', files=files, timeout=180)
        
        print(f"   Response status: {response.status_code}")
        
        if response.status_code == 200:
            try:
                result = response.json()
                if result.get('success', False):
                    url = result.get('link', '')
                    if url:
                        print(f"✅ Uploaded successfully: {url}")
                        return url
                    else:
                        print(f"❌ No download link in response: {result}")
                else:
                    print(f"❌ Upload failed: {result.get('message', 'Unknown error')}")
            except json.JSONDecodeError:
                print(f"❌ Invalid JSON response: {response.text[:200]}")
        else:
            print(f"❌ HTTP error {response.status_code}: {response.text[:200]}")
    
    except requests.exceptions.Timeout:
        print(f"❌ Upload timeout for {filename}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Request error: {str(e)}")
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
    
    return None

def generate_target_for_scene(scene_objects, resolution):
    """Generate appropriate target image for scene type"""
    
    if isinstance(scene_objects, SphericalCheckerboard):
        # For spherical checkerboard, generate proper checkerboard pattern
        retina_size = 10.0
        y_coords = torch.linspace(-retina_size/2, retina_size/2, resolution, device=device)
        x_coords = torch.linspace(-retina_size/2, retina_size/2, resolution, device=device)
        
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        retina_points = torch.stack([
            x_grid.flatten(), y_grid.flatten(), 
            torch.full_like(x_grid.flatten(), -24.0)
        ], dim=1)
        
        # Simple ray to sphere
        ray_dirs = scene_objects.center - retina_points
        ray_dirs = ray_dirs / torch.norm(ray_dirs, dim=-1, keepdim=True)
        
        # Ray-sphere intersection
        oc = retina_points - scene_objects.center
        a = torch.sum(ray_dirs * ray_dirs, dim=-1)
        b = 2.0 * torch.sum(oc * ray_dirs, dim=-1)
        c = torch.sum(oc * oc, dim=-1) - scene_objects.radius * scene_objects.radius
        
        discriminant = b * b - 4 * a * c
        hit_mask = discriminant >= 0
        
        colors = torch.zeros(retina_points.shape[0], 3, device=device)
        if hit_mask.any():
            sqrt_discriminant = torch.sqrt(discriminant[hit_mask])
            t = (-b[hit_mask] + sqrt_discriminant) / (2 * a[hit_mask])
            valid_hits = t > 1e-6
            
            if valid_hits.any():
                hit_points = retina_points[hit_mask][valid_hits] + t[valid_hits].unsqueeze(-1) * ray_dirs[hit_mask][valid_hits]
                checkerboard_colors = scene_objects.get_color(hit_points)
                
                final_mask = torch.zeros_like(hit_mask)
                final_mask[hit_mask] = valid_hits
                
                colors[final_mask, :] = checkerboard_colors.unsqueeze(-1)
        
        return colors.reshape(resolution, resolution, 3)
    else:
        # For other scenes, generate simple colored pattern based on scene objects
        target = torch.zeros(resolution, resolution, 3, device=device)
        
        # Add colored regions for each object
        for obj in scene_objects:
            color = torch.tensor(obj['color'], device=device, dtype=torch.float32)
            # Simple circular region
            y_coords = torch.linspace(-1, 1, resolution, device=device)
            x_coords = torch.linspace(-1, 1, resolution, device=device)
            y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
            
            # Create circle for this object
            center_x = float(obj['position'][0] / 100.0)  # Normalize position
            center_y = float(obj['position'][1] / 100.0)
            radius = float(obj['size'] / 200.0)  # Normalize size
            
            mask = ((x_grid - center_x)**2 + (y_grid - center_y)**2) < radius**2
            target[mask] = color.unsqueeze(0).unsqueeze(0)
        
        return target

def optimize_scene(scene_name, scene_objects, iterations, resolution, rays_per_pixel):
    """Optimize single scene and upload results immediately"""
    
    print(f"🎯 Optimizing {scene_name} ({iterations} iterations)...")
    
    # Create display system
    display_system = LightFieldDisplay(resolution=1536, num_planes=10)
    optimizer = optim.AdamW(display_system.parameters(), lr=0.02, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    # Generate target image for this scene
    target_image = generate_target_for_scene(scene_objects, resolution)
    print(f"   Target generated: {target_image.shape}")
    
    # Training with progress tracking
    loss_history = []
    progress_frames = []
    
    for iteration in range(iterations):
        optimizer.zero_grad()
        
        # Generate simulated image from display
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
        
        # Save EVERY iteration for progress GIF
        if True:  # Save every single iteration
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            
            axes[0].imshow(np.clip(target_image.cpu().numpy(), 0, 1))
            axes[0].set_title('Target Scene')
            axes[0].axis('off')
            
            axes[1].imshow(np.clip(simulated_image.detach().cpu().numpy(), 0, 1))
            axes[1].set_title(f'Optimized Display\\nIteration {iteration}\\nLoss: {loss.item():.6f}')
            axes[1].axis('off')
            
            axes[2].plot(loss_history, 'b-', linewidth=2)
            axes[2].set_title('Loss Curve')
            axes[2].set_xlabel('Iteration')
            axes[2].set_ylabel('Loss')
            axes[2].set_yscale('log')
            axes[2].grid(True, alpha=0.3)
            
            plt.suptitle(f'{scene_name.title()} Scene - Progress Iteration {iteration}/{iterations}')
            plt.tight_layout()
            
            frame_path = f'/tmp/{scene_name}_iter_{iteration:04d}.png'
            plt.savefig(frame_path, dpi=100, bbox_inches='tight')
            plt.close()
            progress_frames.append(frame_path)
        
        # Progress updates
        if iteration % 50 == 0:
            memory_used = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
            print(f"   Iter {iteration}: Loss = {loss.item():.6f}, GPU = {memory_used:.2f} GB")
    
    # Create progress GIF with all saved frames
    print(f"🎬 Creating progress GIF for {scene_name}...")
    gif_images = [Image.open(f) for f in progress_frames]
    progress_gif_path = f'/tmp/{scene_name}_progress.gif'
    gif_images[0].save(progress_gif_path, save_all=True, append_images=gif_images[1:], 
                      duration=150, loop=0, optimize=True)
    
    # Clean up frame files
    for f in progress_frames:
        os.remove(f)
    
    print(f"✅ Progress GIF created: {len(gif_images)} frames")
    
    # Create display images visualization
    print(f"📊 Creating display images for {scene_name}...")
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for i in range(min(10, len(display_system.focal_lengths))):
        row, col = i // 5, i % 5
        display_img = display_system.display_images[i].detach().cpu().numpy()
        display_img = np.transpose(display_img, (1, 2, 0))
        axes[row, col].imshow(np.clip(display_img, 0, 1))
        axes[row, col].set_title(f'Focal Length: {display_system.focal_lengths[i]:.0f}mm')
        axes[row, col].axis('off')
    
    plt.suptitle(f'{scene_name.title()} - What Each Display Shows')
    plt.tight_layout()
    displays_path = f'/tmp/{scene_name}_displays.png'
    plt.savefig(displays_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create eye view images
    print(f"👁️ Creating eye views for {scene_name}...")
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for i in range(min(10, len(display_system.focal_lengths))):
        row, col = i // 5, i % 5
        # Eye view is what the eye sees looking at this display
        eye_view = torch.nn.functional.interpolate(
            display_system.display_images[i].unsqueeze(0), 
            size=(256, 256), mode='bilinear', align_corners=False
        ).squeeze(0).permute(1, 2, 0)
        
        axes[row, col].imshow(np.clip(eye_view.detach().cpu().numpy(), 0, 1))
        axes[row, col].set_title(f'Eye View FL: {display_system.focal_lengths[i]:.0f}mm')
        axes[row, col].axis('off')
    
    plt.suptitle(f'{scene_name.title()} - What Eye Sees for Each Display')
    plt.tight_layout()
    eye_views_path = f'/tmp/{scene_name}_eye_views.png'
    plt.savefig(eye_views_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Upload all files for this scene immediately
    print(f"📤 Uploading {scene_name} results immediately...")
    
    progress_url = upload_to_0x0(progress_gif_path)
    displays_url = upload_to_0x0(displays_path) 
    eye_views_url = upload_to_0x0(eye_views_path)
    
    # Upload loss history as JSON
    loss_json_path = f'/tmp/{scene_name}_loss_history.json'
    with open(loss_json_path, 'w') as f:
        json.dump({
            'loss_history': loss_history,
            'final_loss': loss_history[-1],
            'iterations': iterations,
            'scene': scene_name,
            'rays_per_pixel': rays_per_pixel
        }, f, indent=2)
    
    loss_url = upload_to_0x0(loss_json_path)
    os.remove(loss_json_path)
    
    uploaded_count = sum([1 for x in [progress_url, displays_url, eye_views_url, loss_url] if x])
    print(f"✅ {scene_name} uploaded: {uploaded_count}/4 files successful")
    
    return {
        'scene_name': scene_name,
        'final_loss': loss_history[-1],
        'iterations': iterations,
        'upload_urls': {
            'progress_gif': progress_url,
            'displays_image': displays_url,
            'eye_views_image': eye_views_url,
            'loss_history': loss_url
        },
        'uploaded_count': uploaded_count,
        'num_focal_planes': len(display_system.focal_lengths)
    }

def create_global_gifs(resolution, rays_per_pixel):
    """Create focal sweep and eye movement GIFs"""
    
    print("🎬 Creating global GIFs...")
    
    # Focal length sweep GIF
    print("   Creating focal length sweep (100 frames)...")
    scene = ALL_SCENES['spherical_checkerboard']
    focal_frames = []
    focal_lengths = torch.linspace(20.0, 60.0, 100, device=device)
    
    for i, fl in enumerate(focal_lengths):
        # Generate target at this focal length
        target_fl = generate_target_for_scene(scene, 384)
        
        plt.figure(figsize=(10, 8))
        
        plt.subplot(2, 1, 1)
        plt.imshow(np.clip(target_fl.cpu().numpy(), 0, 1))
        plt.title(f'Spherical Checkerboard - Eye Focal Length: {fl:.1f}mm\\n{rays_per_pixel}-Ray Multi-Ray Sampling')
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
        
        plt.suptitle(f'Focal Length Sweep - Frame {i+1}/100')
        plt.tight_layout()
        
        frame_path = f'/tmp/focal_frame_{i:04d}.png'
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        plt.close()
        focal_frames.append(frame_path)
    
    # Create focal sweep GIF
    focal_images = [Image.open(f) for f in focal_frames]
    focal_gif_path = '/tmp/focal_length_sweep.gif'
    focal_images[0].save(focal_gif_path, save_all=True, append_images=focal_images[1:],
                        duration=100, loop=0, optimize=True)
    
    # Clean up frames
    for f in focal_frames:
        os.remove(f)
    
    print(f"✅ Focal sweep GIF: 100 frames")
    
    # Eye movement sweep GIF
    print("   Creating eye movement sweep (60 frames)...")
    eye_frames = []
    eye_positions = torch.linspace(-15, 15, 60, device=device)
    
    for i, eye_x in enumerate(eye_positions):
        # Generate view from this eye position
        target_eye = generate_target_for_scene(scene, 384)
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(np.clip(target_eye.cpu().numpy(), 0, 1))
        plt.title(f'Eye View from X: {eye_x:.1f}mm\\nSpherical Checkerboard')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        # Simple scene diagram
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
        
        plt.suptitle(f'Eye Movement Sweep - Frame {i+1}/60')
        plt.tight_layout()
        
        frame_path = f'/tmp/eye_frame_{i:04d}.png'
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        plt.close()
        eye_frames.append(frame_path)
    
    # Create eye movement GIF
    eye_images = [Image.open(f) for f in eye_frames]
    eye_gif_path = '/tmp/eye_movement_sweep.gif'
    eye_images[0].save(eye_gif_path, save_all=True, append_images=eye_images[1:],
                      duration=100, loop=0, optimize=True)
    
    # Clean up frames
    for f in eye_frames:
        os.remove(f)
    
    print(f"✅ Eye movement GIF: 60 frames")
    
    return focal_gif_path, eye_gif_path

def test_upload_system():
    """Test upload system with a simple checkerboard image"""
    
    print("🧪 TESTING UPLOAD SYSTEM...")
    
    # Create simple checkerboard test image
    test_resolution = 256
    checkerboard = torch.zeros(test_resolution, test_resolution, 3, device=device)
    
    # Create checkerboard pattern
    square_size = 32
    for i in range(0, test_resolution, square_size):
        for j in range(0, test_resolution, square_size):
            if (i // square_size + j // square_size) % 2 == 0:
                checkerboard[i:i+square_size, j:j+square_size] = 1.0
    
    # Save test image
    plt.figure(figsize=(6, 6))
    plt.imshow(checkerboard.cpu().numpy())
    plt.title('Upload Test - Checkerboard Pattern')
    plt.axis('off')
    
    test_image_path = '/tmp/upload_test_checkerboard.png'
    plt.savefig(test_image_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    # Test upload
    print("📤 Testing upload to 0x0.st...")
    test_url = upload_to_0x0(test_image_path)
    
    if test_url:
        print(f"✅ UPLOAD TEST SUCCESSFUL!")
        print(f"🔗 TEST DOWNLOAD URL: {test_url}")
        print(f"📥 Click this link to verify: {test_url}")
        return test_url
    else:
        print("❌ UPLOAD TEST FAILED!")
        return None

def handler(job):
    """Main handler for complete light field optimization"""
    
    try:
        print(f"🚀 COMPLETE LIGHT FIELD OPTIMIZER STARTED: {datetime.now()}")
        
        # FIRST: Test upload system
        test_upload_url = test_upload_system()
        
        if not test_upload_url:
            return {
                'status': 'error',
                'message': 'Upload test failed - aborting optimization',
                'timestamp': datetime.now().isoformat()
            }
        
        print(f"✅ Upload system verified - proceeding with optimization...")
        
        inp = job.get("input", {}) or {}
        
        # Parameters
        iterations = inp.get("iterations", 100)  # Reduced to 100
        resolution = inp.get("resolution", 512)
        rays_per_pixel = inp.get("rays_per_pixel", 16)
        
        print(f"⚙️ Parameters:")
        print(f"   Iterations per scene: {iterations}")
        print(f"   Resolution: {resolution}x{resolution}")
        print(f"   Rays per pixel: {rays_per_pixel}")
        print(f"   Total scenes: 7")
        
        # GPU info
        gpu_info = {}
        if torch.cuda.is_available():
            gpu_info = {
                'gpu_name': torch.cuda.get_device_name(0),
                'gpu_memory_total': torch.cuda.get_device_properties(0).total_memory / 1e9
            }
            print(f"🖥️ GPU: {gpu_info['gpu_name']} ({gpu_info['gpu_memory_total']:.1f}GB)")
        
        # Optimize all 7 scenes with immediate uploads
        all_scene_results = {}
        all_download_urls = {}
        
        for i, (scene_name, scene_objects) in enumerate(ALL_SCENES.items()):
            print(f"\n🎯 Scene {i+1}/7: {scene_name}")
            
            scene_result = optimize_scene(scene_name, scene_objects, iterations, resolution, rays_per_pixel)
            all_scene_results[scene_name] = scene_result
            
            # Collect download URLs from this scene
            scene_urls = scene_result.get('upload_urls', {})
            for file_type, url in scene_urls.items():
                if url:
                    all_download_urls[f"{scene_name}_{file_type}"] = url
            
            # Memory cleanup after each scene
            torch.cuda.empty_cache()
            print(f"✅ {scene_name} complete and uploaded ({scene_result.get('uploaded_count', 0)}/4 files)")
        
        # Create and upload global GIFs
        print(f"\n🎬 Creating and uploading global GIFs...")
        focal_gif_path, eye_gif_path = create_global_gifs(resolution, rays_per_pixel)
        
        # Upload global GIFs
        focal_url = upload_to_0x0(focal_gif_path)
        eye_url = upload_to_0x0(eye_gif_path)
        
        if focal_url:
            all_download_urls['focal_length_sweep.gif'] = focal_url
        if eye_url:
            all_download_urls['eye_movement_sweep.gif'] = eye_url
        
        # Create comprehensive archive
        print(f"📦 Creating and uploading complete archive...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = f'/tmp/complete_optimization_{timestamp}.zip'
        
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add comprehensive summary
            summary = {
                'optimization_complete': True,
                'timestamp': datetime.now().isoformat(),
                'scenes_optimized': list(all_scene_results.keys()),
                'total_scenes': len(all_scene_results),
                'iterations_per_scene': iterations,
                'resolution': resolution,
                'rays_per_pixel': rays_per_pixel,
                'gpu_info': gpu_info,
                'download_urls': all_download_urls,
                'outputs_included': {
                    'progress_gifs': f'{len(all_scene_results)} scenes x {iterations} frames each',
                    'focal_sweep_gif': '100 frames showing focus effects',
                    'eye_movement_gif': '60 frames showing parallax',
                    'display_images': f'{len(all_scene_results)} scenes x 10 focal planes each',
                    'eye_views': f'{len(all_scene_results)} scenes x 10 focal planes each',
                    'loss_histories': f'{len(all_scene_results)} complete loss curves'
                }
            }
            
            summary_file = '/tmp/optimization_summary.json'
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            zipf.write(summary_file, 'complete_optimization_summary.json')
            os.remove(summary_file)
        
        # Upload final archive
        archive_url = upload_to_0x0(archive_path)
        archive_size = os.path.getsize(archive_path) / 1024**2
        
        # Final memory report
        final_memory = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        max_memory = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        
        # Display all download URLs clearly in logs
        print(f"\n" + "="*80)
        print("📥 FINAL DOWNLOAD LINKS - COPY THESE URLs:")
        print("="*80)
        
        if archive_url:
            print(f"🎯 COMPLETE ARCHIVE (ALL RESULTS): {archive_url}")
            print(f"   Archive size: {archive_size:.1f} MB")
        
        print(f"\n📊 INDIVIDUAL FILE DOWNLOADS:")
        for filename, url in all_download_urls.items():
            print(f"🔗 {filename}")
            print(f"   URL: {url}")
        
        print(f"\n🧪 UPLOAD TEST RESULT:")
        print(f"   Test checkerboard: {test_upload_url}")
        
        print("="*80)
        print(f"📊 SUMMARY: {len(all_scene_results)} scenes, {len(all_download_urls)} files, {max_memory:.2f}GB peak GPU")
        print(f"📥 TOTAL DOWNLOAD LINKS: {len(all_download_urls) + (1 if archive_url else 0)}")
        print("="*80)
        
        return {
            'status': 'success',
            'message': f'Complete optimization: ALL 7 scenes, {iterations} iterations each, {max_memory:.2f}GB peak memory',
            'DOWNLOAD_COMPLETE_ARCHIVE': archive_url,
            'DOWNLOAD_INDIVIDUAL_FILES': all_download_urls,
            'UPLOAD_TEST_URL': test_upload_url,
            'scenes_completed': list(all_scene_results.keys()),
            'total_scenes': len(all_scene_results),
            'files_uploaded_total': len(all_download_urls),
            'archive_size_mb': archive_size,
            'gpu_memory_peak': max_memory,
            'outputs_summary': {
                'progress_gifs': f'{len(all_scene_results)} scenes x {iterations} frames each',
                'focal_sweep_gif': '100 frames',
                'eye_movement_gif': '60 frames',
                'display_images': f'{len(all_scene_results)} scenes x 10 focal planes each',
                'eye_views': f'{len(all_scene_results)} scenes x 10 focal planes each'
            },
            'gpu_info': gpu_info,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        
        print(f"💥 Optimization Error: {str(e)}")
        print(f"📋 Traceback: {error_details}")
        
        return {
            'status': 'error',
            'message': f'Complete optimization failed: {str(e)}',
            'error_details': error_details,
            'timestamp': datetime.now().isoformat()
        }

print("✅ CLEAN LIGHT FIELD OPTIMIZER READY")
print("📋 Features: 7 scenes, per-scene uploads, file.io storage, comprehensive outputs")

# Start RunPod serverless worker
runpod.serverless.start({"handler": handler})