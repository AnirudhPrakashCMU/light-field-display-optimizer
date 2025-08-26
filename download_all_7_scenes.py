#!/usr/bin/env python3
"""
Download All 7-Scene Light Field Optimization Results
Downloads all comprehensive results from the complete optimization
"""

import requests
import os
import json
from pathlib import Path

print("📥 DOWNLOADING COMPLETE 7-SCENE OPTIMIZATION RESULTS")

# Read the new response JSON
with open('response.json', 'r') as f:
    response_data = json.load(f)

output = response_data['output']
all_urls = output['all_download_urls']

# Create organized results directory
results_dir = "complete_7_scene_results"
if os.path.exists(results_dir):
    import shutil
    shutil.rmtree(results_dir)

# Create subdirectories for each scene
scenes = output['scenes_completed']
for scene in scenes:
    os.makedirs(f"{results_dir}/{scene}", exist_ok=True)

os.makedirs(f"{results_dir}/summary", exist_ok=True)

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

# Organize downloads by scene
print("="*70)
downloaded_count = 0
total_files = len(all_urls)

for url_name, url in all_urls.items():
    if 'complete_summary' in url_name:
        # Summary goes to summary folder
        local_path = f"{results_dir}/summary/{url_name.replace('_json', '.json')}"
    else:
        # Parse scene name from URL name
        scene_name = url_name.split('_')[0]
        if scene_name == 'spherical':
            scene_name = 'spherical_checkerboard'
        
        # Determine file type and create appropriate filename
        if 'progress_gif' in url_name:
            filename = 'optimization_progress.gif'
        elif 'optical_focal_sweep' in url_name:
            filename = 'optical_focal_length_sweep.gif'
        elif 'optical_eye_movement' in url_name:
            filename = 'optical_eye_movement_sweep.gif'
        else:
            filename = url_name + '.file'
        
        local_path = f"{results_dir}/{scene_name}/{filename}"
    
    success = download_file(url, local_path)
    if success:
        downloaded_count += 1

# Save comprehensive metadata
metadata = {
    'optimization_complete': True,
    'timestamp': output['timestamp'],
    'execution_time_seconds': response_data['executionTime'],
    'total_scenes': output['total_scenes'],
    'scenes_completed': output['scenes_completed'],
    'scene_final_losses': output['scene_results'],
    'optimization_specs': output['optimization_specs'],
    'download_urls': all_urls,
    'files_structure': {
        'total_files': total_files,
        'per_scene_files': 3,
        'summary_files': 1,
        'scene_outputs': {
            'optimization_progress.gif': 'Shows optimization convergence (3 frames)',
            'optical_focal_length_sweep.gif': 'Eye view through optical system with focal sweep (15 frames)', 
            'optical_eye_movement_sweep.gif': 'Eye view through optical system with eye movement (10 frames)'
        }
    }
}

metadata_path = f"{results_dir}/optimization_metadata.json"
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print("="*70)
print(f"📊 DOWNLOAD SUMMARY:")
print(f"✅ Successfully downloaded: {downloaded_count}/{total_files} files")
print(f"📁 All files organized in: {results_dir}/")
print(f"📋 Metadata saved to: {metadata_path}")

# Display organized structure
print(f"\n📁 COMPLETE RESULTS STRUCTURE:")
for root, dirs, files in os.walk(results_dir):
    level = root.replace(results_dir, '').count(os.sep)
    indent = '  ' * level
    folder_name = os.path.basename(root) if level > 0 else results_dir
    print(f"{indent}{folder_name}/")
    
    subindent = '  ' * (level + 1)
    for file in sorted(files):
        file_path = os.path.join(root, file)
        file_size = os.path.getsize(file_path) / 1024**2
        print(f"{subindent}{file} ({file_size:.1f} MB)")

print(f"\n📊 OPTIMIZATION SUMMARY:")
print(f"   Total scenes: {output['total_scenes']}")
print(f"   Iterations per scene: {output['optimization_specs']['iterations_per_scene']}")
print(f"   Resolution: {output['optimization_specs']['resolution']}x{output['optimization_specs']['resolution']}")
print(f"   Execution time: {response_data['executionTime']/1000:.1f} seconds")

print(f"\n📈 FINAL LOSSES BY SCENE:")
for scene, data in output['scene_results'].items():
    print(f"   {scene}: {data['final_loss']:.8f}")

print(f"\n✅ COMPLETE 7-SCENE OPTIMIZATION RESULTS DOWNLOADED AND ORGANIZED!")