#!/usr/bin/env python3
"""
Download All Light Field Optimization Results
Downloads all optimization results from catbox.moe URLs and organizes them
"""

import requests
import os
from pathlib import Path
import json

print("📥 DOWNLOADING ALL LIGHT FIELD OPTIMIZATION RESULTS")
print("="*60)

# All download URLs from the optimization logs
download_urls = {
    'verification/upload_test_checkerboard.png': 'https://files.catbox.moe/m08e7h.png',
    
    # Basic scene
    'basic/progress.gif': 'https://files.catbox.moe/g33hgg.gif',
    'basic/displays.png': 'https://files.catbox.moe/0beq5n.png', 
    'basic/eye_views.png': 'https://files.catbox.moe/4axm6p.png',
    'basic/loss_history.json': 'https://files.catbox.moe/3srifa.json',
    
    # Complex scene
    'complex/progress.gif': 'https://files.catbox.moe/pbql4x.gif',
    'complex/displays.png': 'https://files.catbox.moe/lk19sp.png',
    'complex/eye_views.png': 'https://files.catbox.moe/v1iche.png',
    'complex/loss_history.json': 'https://files.catbox.moe/hmn7st.json',
    
    # Stick figure scene
    'stick_figure/progress.gif': 'https://files.catbox.moe/drfkax.gif',
    'stick_figure/displays.png': 'https://files.catbox.moe/k55fec.png',
    'stick_figure/eye_views.png': 'https://files.catbox.moe/jswtvk.png',
    'stick_figure/loss_history.json': 'https://files.catbox.moe/61ywye.json',
    
    # Layered scene
    'layered/progress.gif': 'https://files.catbox.moe/v6nt07.gif',
    'layered/displays.png': 'https://files.catbox.moe/m5e5qs.png',
    'layered/eye_views.png': 'https://files.catbox.moe/bhobvm.png',
    'layered/loss_history.json': 'https://files.catbox.moe/wsn66e.json',
    
    # Office scene  
    'office/progress.gif': 'https://files.catbox.moe/g8e0y3.gif',
    'office/displays.png': 'https://files.catbox.moe/nyj72e.png',
    'office/eye_views.png': 'https://files.catbox.moe/f41il4.png',
    'office/loss_history.json': 'https://files.catbox.moe/w9nktu.json',
    
    # Nature scene
    'nature/progress.gif': 'https://files.catbox.moe/zx12j8.gif',
    'nature/displays.png': 'https://files.catbox.moe/6joxmt.png',
    'nature/eye_views.png': 'https://files.catbox.moe/blivxh.png',
    'nature/loss_history.json': 'https://files.catbox.moe/qi03c4.json',
    
    # Spherical checkerboard scene
    'spherical_checkerboard/progress.gif': 'https://files.catbox.moe/b9k5el.gif',
    'spherical_checkerboard/displays.png': 'https://files.catbox.moe/c5ql62.png',
    'spherical_checkerboard/eye_views.png': 'https://files.catbox.moe/azw4n0.png',
    'spherical_checkerboard/loss_history.json': 'https://files.catbox.moe/y4t3oq.json'
}

def download_file(url, local_path):
    """Download a single file with error handling"""
    
    try:
        print(f"📤 Downloading {os.path.basename(local_path)}...")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # Download with headers to avoid blocking
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=60, stream=True)
        
        if response.status_code == 200:
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            file_size = os.path.getsize(local_path) / 1024**2
            print(f"✅ Downloaded: {os.path.basename(local_path)} ({file_size:.1f} MB)")
            return True
        else:
            print(f"❌ Failed: {os.path.basename(local_path)} (HTTP {response.status_code})")
            return False
            
    except Exception as e:
        print(f"❌ Error downloading {os.path.basename(local_path)}: {e}")
        return False

def main():
    """Download all optimization results"""
    
    base_dir = "downloaded_results"
    
    # Remove existing directory and recreate
    import shutil
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(base_dir, exist_ok=True)
    
    # Download all files
    downloaded_count = 0
    failed_count = 0
    
    for relative_path, url in download_urls.items():
        local_path = os.path.join(base_dir, relative_path)
        
        success = download_file(url, local_path)
        
        if success:
            downloaded_count += 1
        else:
            failed_count += 1
    
    print(f"\n" + "="*60)
    print(f"📊 DOWNLOAD SUMMARY:")
    print(f"✅ Successfully downloaded: {downloaded_count} files")
    print(f"❌ Failed downloads: {failed_count} files") 
    print(f"📁 All files saved to: {base_dir}/")
    print("="*60)
    
    # Create download summary
    summary = {
        'download_date': str(datetime.now()),
        'total_files': len(download_urls),
        'downloaded_successfully': downloaded_count,
        'failed_downloads': failed_count,
        'download_directory': base_dir,
        'scenes': ['basic', 'complex', 'stick_figure', 'layered', 'office', 'nature', 'spherical_checkerboard'],
        'files_per_scene': ['progress.gif', 'displays.png', 'eye_views.png', 'loss_history.json'],
        'optimization_specs': {
            'iterations_per_scene': 100,
            'resolution': '512x512',
            'rays_per_pixel': 16,
            'display_resolution': '1536x1536',
            'focal_planes_per_scene': 10,
            'gpu': 'NVIDIA A100-SXM4-80GB',
            'peak_memory': '2.12 GB',
            'total_runtime': '~10 minutes'
        }
    }
    
    summary_file = os.path.join(base_dir, 'download_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📋 Download summary saved: {summary_file}")
    
    # List directory structure
    print(f"\n📁 DOWNLOADED DIRECTORY STRUCTURE:")
    for root, dirs, files in os.walk(base_dir):
        level = root.replace(base_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            file_size = os.path.getsize(os.path.join(root, file)) / 1024**2
            print(f"{subindent}{file} ({file_size:.1f} MB)")

if __name__ == "__main__":
    from datetime import datetime
    main()