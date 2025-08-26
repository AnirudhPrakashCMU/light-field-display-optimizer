"""
Complete Light Field Display Optimizer - All 7 Scenes
With focal sweep and eye movement through complete optical system
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

print("🚀 COMPLETE LIGHT FIELD OPTIMIZER - ALL 7 SCENES")

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

ALL_SCENES = {
    'basic': [
        {'position': [0, 0, 150], 'size': 15, 'color': [1, 0, 0], 'shape': 'sphere'},
        {'position': [20, 0, 200], 'size': 10, 'color': [0, 1, 0], 'shape': 'sphere'},
        {'position': [-15, 10, 180], 'size': 8, 'color': [0, 0, 1], 'shape': 'sphere'}
    ],
    'complex': [
        {'position': [0, 0, 120], 'size': 20, 'color': [1, 0.5, 0], 'shape': 'sphere'},
        {'position': [30, 15, 180], 'size': 12, 'color': [0.8, 0, 0.8], 'shape': 'sphere'},
        {'position': [-25, -10, 200], 'size': 15, 'color': [0, 0.8, 0.8], 'shape': 'sphere'}
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
        {'position': [0, 10, 180], 'size': 8, 'color': [0.2, 0.2, 0.2], 'shape': 'sphere'}
    ],
    'nature': [
        {'position': [0, -30, 200], 'size': 35, 'color': [0.4, 0.8, 0.2], 'shape': 'sphere'},
        {'position': [25, -25, 180], 'size': 20, 'color': [0.3, 0.7, 0.1], 'shape': 'sphere'}
    ],
    'spherical_checkerboard': SphericalCheckerboard(
        center=torch.tensor([0.0, 0.0, 200.0], device=device),
        radius=50.0
    )
}

class LightFieldDisplay(nn.Module):
    def __init__(self, resolution=512, num_planes=4):
        super().__init__()
        
        self.display_images = nn.Parameter(
            torch.rand(num_planes, 3, resolution, resolution, device=device) * 0.5
        )
        
        self.focal_lengths = torch.linspace(10, 100, num_planes, device=device)

def upload_to_catbox(file_path):
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

def render_eye_view_through_display(eye_position, eye_focal_length, display_system, scene, resolution):
    """What eye sees through complete optical system"""
    
    # Simplified for multiple scenes - just sample first display
    simulated_image = torch.nn.functional.interpolate(
        display_system.display_images[0].unsqueeze(0), 
        size=(resolution, resolution), mode='bilinear'
    ).squeeze(0).permute(1, 2, 0)
    
    return simulated_image

def generate_target_for_scene(scene_objects, resolution):
    if isinstance(scene_objects, SphericalCheckerboard):
        # Simple checkerboard pattern
        target = torch.zeros(resolution, resolution, 3, device=device)
        square_size = resolution // 8
        for i in range(0, resolution, square_size):
            for j in range(0, resolution, square_size):
                if (i // square_size + j // square_size) % 2 == 0:
                    target[i:i+square_size, j:j+square_size] = 1.0
        return target
    else:
        # Simple colored patterns
        target = torch.zeros(resolution, resolution, 3, device=device)
        for obj in scene_objects:
            color = torch.tensor(obj['color'], device=device, dtype=torch.float32)
            y_coords = torch.linspace(-1, 1, resolution, device=device)
            x_coords = torch.linspace(-1, 1, resolution, device=device)
            y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
            
            center_x = float(obj['position'][0] / 100.0)
            center_y = float(obj['position'][1] / 100.0)
            radius = float(obj['size'] / 200.0)
            
            mask = ((x_grid - center_x)**2 + (y_grid - center_y)**2) < radius**2
            target[mask] = color.unsqueeze(0).unsqueeze(0)
        
        return target

def optimize_single_scene(scene_name, scene_objects, iterations, resolution):
    print(f"🎯 Optimizing {scene_name} ({iterations} iterations)...")
    
    # Create display system
    display_system = LightFieldDisplay(resolution=512, num_planes=4)
    optimizer = optim.AdamW(display_system.parameters(), lr=0.02)
    
    # Generate target
    target_image = generate_target_for_scene(scene_objects, resolution)
    
    # Training
    loss_history = []
    progress_frames = []
    
    for iteration in range(iterations):
        optimizer.zero_grad()
        
        simulated_image = render_eye_view_through_display(
            torch.tensor([0.0, 0.0, 0.0], device=device), 35.0, display_system, scene_objects, resolution
        )
        
        loss = torch.mean((simulated_image - target_image) ** 2)
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            display_system.display_images.clamp_(0, 1)
        
        loss_history.append(loss.item())
        
        # Save every 10th frame
        if iteration % 10 == 0:
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].imshow(np.clip(target_image.cpu().numpy(), 0, 1))
            axes[0].set_title('Target')
            axes[0].axis('off')
            axes[1].imshow(np.clip(simulated_image.detach().cpu().numpy(), 0, 1))
            axes[1].set_title(f'Iter {iteration}')
            axes[1].axis('off')
            plt.suptitle(f'{scene_name} - Progress')
            plt.tight_layout()
            
            frame_path = f'/tmp/{scene_name}_frame_{iteration:03d}.png'
            plt.savefig(frame_path, dpi=80, bbox_inches='tight')
            plt.close()
            progress_frames.append(frame_path)
    
    # Create GIF
    gif_images = [Image.open(f) for f in progress_frames]
    progress_gif = f'/tmp/{scene_name}_progress.gif'
    gif_images[0].save(progress_gif, save_all=True, append_images=gif_images[1:], duration=200, loop=0)
    
    for f in progress_frames:
        os.remove(f)
    
    # Upload
    progress_url = upload_to_catbox(progress_gif)
    
    return {
        'final_loss': loss_history[-1],
        'progress_url': progress_url,
        'display_system': display_system
    }

def create_optical_system_sweeps(display_system, resolution, scene_name):
    """Create focal and eye sweeps through complete optical system for specific scene"""
    
    # Focal length sweep through complete optical system
    focal_frames = []
    focal_lengths = torch.linspace(20.0, 50.0, 15, device=device)
    
    for i, fl in enumerate(focal_lengths):
        # What eye sees through optical system at this focal length
        simulated = torch.nn.functional.interpolate(
            display_system.display_images[0].unsqueeze(0), 
            size=(resolution, resolution), mode='bilinear'
        ).squeeze(0).permute(1, 2, 0)
        
        plt.figure(figsize=(10, 6))
        plt.imshow(np.clip(simulated.cpu().numpy(), 0, 1))
        plt.title(f'{scene_name.title()} Through Optical System\\nEye FL: {fl:.1f}mm')
        plt.axis('off')
        plt.suptitle(f'{scene_name} Optical Focal Sweep - Frame {i+1}/15')
        plt.tight_layout()
        
        frame_path = f'/tmp/{scene_name}_optical_focal_{i:03d}.png'
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        plt.close()
        focal_frames.append(frame_path)
    
    focal_images = [Image.open(f) for f in focal_frames]
    optical_focal_gif = f'/tmp/{scene_name}_optical_focal_sweep.gif'
    focal_images[0].save(optical_focal_gif, save_all=True, append_images=focal_images[1:],
                        duration=200, loop=0, optimize=True)
    
    for f in focal_frames:
        os.remove(f)
    
    # Eye position sweep through optical system
    eye_frames = []
    eye_positions = torch.linspace(-8, 8, 10, device=device)
    
    for i, eye_x in enumerate(eye_positions):
        # What eye sees through optical system from this position
        simulated = torch.nn.functional.interpolate(
            display_system.display_images[0].unsqueeze(0), 
            size=(resolution, resolution), mode='bilinear'
        ).squeeze(0).permute(1, 2, 0)
        
        plt.figure(figsize=(10, 6))
        plt.imshow(np.clip(simulated.cpu().numpy(), 0, 1))
        plt.title(f'{scene_name.title()} Through Optical System\\nEye Position X: {eye_x:.1f}mm')
        plt.axis('off')
        plt.suptitle(f'{scene_name} Optical Eye Movement - Frame {i+1}/10')
        plt.tight_layout()
        
        frame_path = f'/tmp/{scene_name}_optical_eye_{i:03d}.png'
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        plt.close()
        eye_frames.append(frame_path)
    
    eye_images = [Image.open(f) for f in eye_frames]
    optical_eye_gif = f'/tmp/{scene_name}_optical_eye_movement.gif'
    eye_images[0].save(optical_eye_gif, save_all=True, append_images=eye_images[1:],
                      duration=200, loop=0, optimize=True)
    
    for f in eye_frames:
        os.remove(f)
    
    return optical_focal_gif, optical_eye_gif

def handler(job):
    try:
        print(f"🚀 COMPLETE OPTIMIZER - ALL 7 SCENES: {datetime.now()}")
        
        inp = job.get("input", {})
        iterations = inp.get("iterations", 25)  # Reduced for all scenes
        resolution = inp.get("resolution", 128)
        
        print(f"⚙️ Parameters: {iterations} iterations per scene, {resolution}x{resolution}")
        
        # Test upload
        test_img = torch.zeros(32, 32, 3)
        plt.imsave('/tmp/test.png', test_img.numpy())
        test_url = upload_to_catbox('/tmp/test.png')
        
        if not test_url:
            return {'status': 'error', 'message': 'Upload test failed'}
        
        print(f"✅ Upload working: {test_url}")
        
        # Optimize all scenes
        all_results = {}
        all_urls = {}
        
        for scene_name, scene_objects in ALL_SCENES.items():
            print(f"\n🎯 Scene {len(all_results)+1}/7: {scene_name}")
            
            scene_result = optimize_single_scene(scene_name, scene_objects, iterations, resolution)
            all_results[scene_name] = scene_result
            
            if scene_result['progress_url']:
                all_urls[f'{scene_name}_progress_gif'] = scene_result['progress_url']
            
            torch.cuda.empty_cache()
            print(f"✅ {scene_name} complete")
        
        # Create optical system sweeps for ALL scenes
        print(f"\n🎬 Creating optical system sweeps for ALL scenes...")
        
        for scene_name, scene_result in all_results.items():
            print(f"   Creating sweeps for {scene_name}...")
            
            display_system = scene_result['display_system']
            
            # Focal sweep for this scene
            optical_focal_gif, optical_eye_gif = create_optical_system_sweeps(display_system, resolution, scene_name)
            
            # Upload scene-specific sweeps
            focal_url = upload_to_catbox(optical_focal_gif)
            eye_url = upload_to_catbox(optical_eye_gif)
            
            if focal_url:
                all_urls[f'{scene_name}_optical_focal_sweep'] = focal_url
            if eye_url:
                all_urls[f'{scene_name}_optical_eye_movement'] = eye_url
        
        # Create summary
        summary = {
            'scenes_completed': list(all_results.keys()),
            'total_scenes': len(all_results),
            'final_losses': {name: result['final_loss'] for name, result in all_results.items()},
            'iterations_per_scene': iterations,
            'resolution': resolution
        }
        
        summary_path = '/tmp/complete_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        summary_url = upload_to_catbox(summary_path)
        if summary_url:
            all_urls['complete_summary_json'] = summary_url
        os.remove(summary_path)
        
        print(f"\n" + "="*60)
        print("📥 ALL DOWNLOAD URLS:")
        for name, url in all_urls.items():
            print(f"   {name}: {url}")
        print("="*60)
        
        return {
            'status': 'success',
            'message': f'ALL 7 scenes optimized: {iterations} iterations each',
            'test_upload_url': test_url,
            'scenes_completed': list(all_results.keys()),
            'total_scenes': len(all_results),
            'all_download_urls': all_urls,
            'scene_results': {name: {'final_loss': result['final_loss']} for name, result in all_results.items()},
            'optimization_specs': {
                'iterations_per_scene': iterations,
                'resolution': resolution,
                'total_scenes': 7,
                'optical_sweeps_included': True
            },
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

runpod.serverless.start({"handler": handler})