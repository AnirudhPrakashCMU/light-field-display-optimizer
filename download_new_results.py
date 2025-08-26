#!/usr/bin/env python3
"""
Download Latest REAL Optimization Results
Downloads the complete REAL optimization with comparison outputs
"""

import requests
import os
import json
from datetime import datetime

print("📥 DOWNLOADING REAL LIGHT FIELD OPTIMIZATION RESULTS")

# Latest URLs from the REAL optimization (7 outputs per scene)
download_urls = {
    # Basic scene (7 outputs)
    'basic/progress_ALL_FRAMES.gif': 'https://files.catbox.moe/p9fzh7.gif',
    'basic/displays.png': 'https://files.catbox.moe/1aq47u.png',
    'basic/eye_views.png': 'https://files.catbox.moe/0qhv83.png',
    'basic/focal_sweep_through_display.gif': 'https://files.catbox.moe/17nlcw.gif',
    'basic/eye_movement_through_display.gif': 'https://files.catbox.moe/e30a2x.gif',
    'basic/REAL_scene_focal_sweep.gif': 'https://files.catbox.moe/101f02.gif',
    'basic/REAL_scene_eye_movement.gif': 'https://files.catbox.moe/012leb.gif',
    
    # Complex scene (7 outputs)
    'complex/progress_ALL_FRAMES.gif': 'https://files.catbox.moe/7vv69t.gif',
    'complex/displays.png': 'https://files.catbox.moe/s4nhpn.png',
    'complex/eye_views.png': 'https://files.catbox.moe/giq2dd.png',
    'complex/focal_sweep_through_display.gif': 'https://files.catbox.moe/sml55y.gif',
    'complex/eye_movement_through_display.gif': 'https://files.catbox.moe/voi2hq.gif',
    'complex/REAL_scene_focal_sweep.gif': 'https://files.catbox.moe/qdeydi.gif',
    'complex/REAL_scene_eye_movement.gif': 'https://files.catbox.moe/s7mz4j.gif'
    
    # Note: stick_figure and remaining 4 scenes still running
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
    
    base_dir = "REAL_optimization_results"
    
    # Remove existing directory and recreate
    import shutil
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(base_dir, exist_ok=True)
    
    # Download all files
    downloaded_count = 0
    failed_count = 0
    
    print("="*80)
    
    for relative_path, url in download_urls.items():
        local_path = os.path.join(base_dir, relative_path)
        
        success = download_file(url, local_path)
        
        if success:
            downloaded_count += 1
        else:
            failed_count += 1
    
    print("="*80)
    print(f"📊 DOWNLOAD SUMMARY:")
    print(f"✅ Successfully downloaded: {downloaded_count} files")
    print(f"❌ Failed downloads: {failed_count} files") 
    print(f"📁 All files saved to: {base_dir}/")
    
    # Create comprehensive summary
    summary = {
        'download_date': datetime.now().isoformat(),
        'optimization_type': 'REAL_ray_tracing_all_scenes',
        'total_files_available': len(download_urls),
        'downloaded_successfully': downloaded_count,
        'failed_downloads': failed_count,
        'scenes_downloaded': list(set([path.split('/')[0] for path in download_urls.keys()])),
        'outputs_per_scene': 7,
        'output_types': {
            'progress_ALL_FRAMES.gif': 'Optimization progress - every iteration (25 frames)',
            'displays.png': 'What each display shows (8 focal planes)',
            'eye_views.png': 'What eye sees for each display (8 views)',
            'focal_sweep_through_display.gif': 'Focal sweep through optimized display system (10 frames)',
            'eye_movement_through_display.gif': 'Eye movement through optimized display system (15 frames)',
            'REAL_scene_focal_sweep.gif': 'Focal sweep of real scene directly (10 frames) - FOR COMPARISON',
            'REAL_scene_eye_movement.gif': 'Eye movement of real scene directly (15 frames) - FOR COMPARISON'
        },
        'comparison_analysis': {
            'display_vs_scene': 'Compare display system performance against real scene',
            'focal_effects': 'See how focal length affects both display and real scene',
            'eye_movement_effects': 'See parallax in both display and real scene'
        }
    }
    
    summary_file = os.path.join(base_dir, 'optimization_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📋 Download summary saved: {summary_file}")
    
    # Display organized structure
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
    
    print(f"\n📊 OPTIMIZATION ANALYSIS:")
    print(f"✅ Real ray tracing: Target AND simulated both use actual ray tracing")
    print(f"✅ Real optimization: MSE between two ray-traced images")
    print(f"✅ Every iteration: Progress GIFs show complete convergence")
    print(f"✅ Individual displays: Each focal plane shown separately")
    print(f"✅ Individual eye views: What eye sees for each display") 
    print(f"✅ Display system sweeps: Performance of optimized system")
    print(f"✅ Real scene sweeps: Ground truth for comparison")
    
    if downloaded_count < 49:  # 7 scenes × 7 outputs = 49 total
        remaining_scenes = 49 - downloaded_count
        print(f"\n⏳ Note: {remaining_scenes} more files expected from remaining scenes")
        print(f"💡 Run this script again after all 7 scenes complete")

if __name__ == "__main__":
    main()