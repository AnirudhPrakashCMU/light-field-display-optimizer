#!/usr/bin/env python3
"""
Download All Light Field Optimization Results
Downloads all 35 results from the complete optimization
"""

import requests
import os
import json
from pathlib import Path

print("📥 DOWNLOADING ALL LIGHT FIELD OPTIMIZATION RESULTS")

# All URLs from the running optimization
download_urls = {
    # Basic scene
    'basic/progress_all_frames.gif': 'https://files.catbox.moe/xkb75g.gif',
    'basic/displays.png': 'https://files.catbox.moe/ve62ta.png',
    'basic/eye_views.png': 'https://files.catbox.moe/4bdypk.png',
    'basic/focal_sweep.gif': 'https://files.catbox.moe/soa2cu.gif',
    'basic/eye_movement.gif': 'https://files.catbox.moe/rsnefl.gif',
    
    # Complex scene
    'complex/progress_all_frames.gif': 'https://files.catbox.moe/50lnml.gif',
    'complex/displays.png': 'https://files.catbox.moe/2wpd6z.png',
    'complex/eye_views.png': 'https://files.catbox.moe/sqysdt.png',
    'complex/focal_sweep.gif': 'https://files.catbox.moe/he04fk.gif',
    'complex/eye_movement.gif': 'https://files.catbox.moe/7m6ytj.gif',
    
    # Stick figure scene
    'stick_figure/progress_all_frames.gif': 'https://files.catbox.moe/zfev28.gif',
    'stick_figure/displays.png': 'https://files.catbox.moe/dz8jlh.png',
    'stick_figure/eye_views.png': 'https://files.catbox.moe/eu0em6.png',
    'stick_figure/focal_sweep.gif': 'https://files.catbox.moe/8rd50f.gif',
    'stick_figure/eye_movement.gif': 'https://files.catbox.moe/jovef2.gif'
    
    # Note: Add remaining scenes (layered, office, nature, spherical_checkerboard) when they complete
}

def download_file(url, local_path):
    """Download file with progress"""
    try:
        print(f"📤 Downloading {os.path.basename(local_path)}...")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=120, stream=True)
        
        if response.status_code == 200:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
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
    """Download all available results"""
    
    base_dir = "complete_optimization_results"
    
    # Remove existing directory and recreate
    import shutil
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(base_dir, exist_ok=True)
    
    # Download all files
    downloaded_count = 0
    failed_count = 0
    
    print("="*70)
    
    for relative_path, url in download_urls.items():
        local_path = os.path.join(base_dir, relative_path)
        
        success = download_file(url, local_path)
        
        if success:
            downloaded_count += 1
        else:
            failed_count += 1
    
    print("="*70)
    print(f"📊 DOWNLOAD SUMMARY:")
    print(f"✅ Successfully downloaded: {downloaded_count} files")
    print(f"❌ Failed downloads: {failed_count} files") 
    print(f"📁 All files saved to: {base_dir}/")
    
    # Create summary
    summary = {
        'download_date': str(datetime.now()),
        'total_files': len(download_urls),
        'downloaded_successfully': downloaded_count,
        'failed_downloads': failed_count,
        'optimization_status': 'partial' if downloaded_count < 35 else 'complete',
        'scenes_downloaded': list(set([path.split('/')[0] for path in download_urls.keys()])),
        'remaining_scenes': ['layered', 'office', 'nature', 'spherical_checkerboard'] if downloaded_count < 35 else []
    }
    
    summary_file = os.path.join(base_dir, 'download_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📋 Download summary saved: {summary_file}")
    
    # List directory structure
    print(f"\n📁 DOWNLOADED DIRECTORY STRUCTURE:")
    for root, dirs, files in os.walk(base_dir):
        level = root.replace(base_dir, '').count(os.sep)
        indent = '  ' * level
        folder_name = os.path.basename(root) if level > 0 else base_dir
        print(f"{indent}{folder_name}/")
        
        subindent = '  ' * (level + 1)
        for file in sorted(files):
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path) / 1024**2
            print(f"{subindent}{file} ({file_size:.1f} MB)")
    
    if downloaded_count < 35:
        print(f"\n⏳ Note: Only {downloaded_count}/35 files downloaded (optimization still running)")
        print(f"💡 Run this script again after all 7 scenes complete to get remaining files")

if __name__ == "__main__":
    main()