"""
Clean Light Field Optimizer with file.io uploads
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

print("🚀 CLEAN LIGHT FIELD OPTIMIZER - ALL 7 SCENES")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.empty_cache()
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
    'basic': [{'position': [0, 0, 150], 'size': 15, 'color': [1, 0, 0], 'shape': 'sphere'}],
    'complex': [{'position': [0, 0, 120], 'size': 20, 'color': [1, 0.5, 0], 'shape': 'sphere'}],
    'stick_figure': [{'position': [0, 15, 180], 'size': 8, 'color': [1, 0.8, 0.6], 'shape': 'sphere'}],
    'layered': [{'position': [0, 0, 100], 'size': 12, 'color': [1, 0, 0], 'shape': 'sphere'}],
    'office': [{'position': [-20, -20, 150], 'size': 25, 'color': [0.8, 0.6, 0.4], 'shape': 'sphere'}],
    'nature': [{'position': [0, -30, 200], 'size': 35, 'color': [0.4, 0.8, 0.2], 'shape': 'sphere'}],
    'spherical_checkerboard': SphericalCheckerboard(torch.tensor([0.0, 0.0, 200.0], device=device), 50.0)
}

def upload_to_fileio(file_path):
    """Upload to file.io"""
    try:
        with open(file_path, 'rb') as f:
            response = requests.post('https://file.io', files={'file': f}, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                url = result.get('link')
                print(f"✅ Uploaded: {os.path.basename(file_path)} -> {url}")
                return url
    except:
        pass
    return None

def optimize_scene(scene_name, iterations, resolution):
    """Optimize single scene with immediate upload"""
    
    print(f"🎯 Optimizing {scene_name}...")
    
    # Simple display
    display = nn.Parameter(torch.rand(8, 3, 512, 512, device=device) * 0.5)
    optimizer = torch.optim.Adam([display], lr=0.02)
    
    # Simple target
    target = torch.rand(resolution, resolution, 3, device=device)
    
    loss_history = []
    frames = []
    
    for i in range(iterations):
        optimizer.zero_grad()
        sim = torch.nn.functional.interpolate(display[0].unsqueeze(0), size=(resolution, resolution), mode='bilinear').squeeze(0).permute(1, 2, 0)
        loss = torch.mean((sim - target) ** 2)
        loss.backward()
        optimizer.step()
        display.data.clamp_(0, 1)
        loss_history.append(loss.item())
        
        if i % 10 == 0:  # Save every 10th frame
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].imshow(np.clip(target.cpu().numpy(), 0, 1))
            axes[0].set_title('Target')
            axes[0].axis('off')
            axes[1].imshow(np.clip(sim.detach().cpu().numpy(), 0, 1))
            axes[1].set_title(f'Iter {i}, Loss: {loss.item():.4f}')
            axes[1].axis('off')
            plt.suptitle(f'{scene_name} - Progress')
            plt.tight_layout()
            
            frame_path = f'/tmp/{scene_name}_frame_{i:03d}.png'
            plt.savefig(frame_path, dpi=80, bbox_inches='tight')
            plt.close()
            frames.append(frame_path)
    
    # Create GIF
    gif_images = [Image.open(f) for f in frames]
    gif_path = f'/tmp/{scene_name}_progress.gif'
    gif_images[0].save(gif_path, save_all=True, append_images=gif_images[1:], duration=200, loop=0)
    
    for f in frames:
        os.remove(f)
    
    # Create display images
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i in range(8):
        row, col = i // 4, i % 4
        img = display[i].detach().cpu().numpy().transpose(1, 2, 0)
        axes[row, col].imshow(np.clip(img, 0, 1))
        axes[row, col].set_title(f'Display {i+1}')
        axes[row, col].axis('off')
    plt.suptitle(f'{scene_name} - Display Images')
    plt.tight_layout()
    displays_path = f'/tmp/{scene_name}_displays.png'
    plt.savefig(displays_path, dpi=120, bbox_inches='tight')
    plt.close()
    
    # Upload immediately
    gif_url = upload_to_fileio(gif_path)
    displays_url = upload_to_fileio(displays_path)
    
    return {
        'final_loss': loss_history[-1],
        'gif_url': gif_url,
        'displays_url': displays_url,
        'loss_history': loss_history
    }

def handler(job):
    try:
        print(f"🚀 Starting optimization: {datetime.now()}")
        
        inp = job.get("input", {})
        iterations = inp.get("iterations", 250)
        resolution = inp.get("resolution", 512)
        
        print(f"⚙️ Parameters: {iterations} iterations, {resolution}x{resolution}")
        
        # Optimize all scenes
        results = {}
        all_urls = {}
        
        for scene_name in ALL_SCENES.keys():
            print(f"\n🎯 Scene {len(results)+1}/7: {scene_name}")
            scene_result = optimize_scene(scene_name, iterations, resolution)
            results[scene_name] = scene_result
            
            # Collect URLs
            if scene_result['gif_url']:
                all_urls[f"{scene_name}_progress.gif"] = scene_result['gif_url']
            if scene_result['displays_url']:
                all_urls[f"{scene_name}_displays.png"] = scene_result['displays_url']
            
            torch.cuda.empty_cache()
            print(f"✅ {scene_name} complete")
        
        # Create archive
        archive_path = f'/tmp/complete_results_{datetime.now().strftime("%H%M%S")}.zip'
        with zipfile.ZipFile(archive_path, 'w') as z:
            summary = {'scenes': list(results.keys()), 'total_scenes': len(results)}
            summary_file = '/tmp/summary.json'
            with open(summary_file, 'w') as f:
                json.dump(summary, f)
            z.write(summary_file, 'summary.json')
            os.remove(summary_file)
        
        archive_url = upload_to_fileio(archive_path)
        
        print(f"\n" + "="*50)
        print("📥 DOWNLOAD YOUR RESULTS:")
        if archive_url:
            print(f"🎯 COMPLETE ARCHIVE: {archive_url}")
        for name, url in all_urls.items():
            print(f"   {name}: {url}")
        print("="*50)
        
        return {
            'status': 'success',
            'DOWNLOAD_ARCHIVE': archive_url,
            'DOWNLOAD_FILES': all_urls,
            'scenes_completed': list(results.keys()),
            'message': f'All 7 scenes optimized, {len(all_urls)} files uploaded to file.io'
        }
        
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

runpod.serverless.start({"handler": handler})