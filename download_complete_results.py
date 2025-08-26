#!/usr/bin/env python3
"""
Download Complete Light Field Optimization Results
Downloads all results from the comprehensive optimization response
"""

import requests
import os
import json
from pathlib import Path

print("📥 DOWNLOADING COMPLETE LIGHT FIELD OPTIMIZATION RESULTS")

# Read the response JSON
with open('response.json', 'r') as f:
    response_data = json.load(f)

output = response_data['output']
all_urls = output['all_download_urls']

# Create results directory
results_dir = "optimization_results"
if os.path.exists(results_dir):
    import shutil
    shutil.rmtree(results_dir)
os.makedirs(results_dir, exist_ok=True)

# Download mapping with organized names
downloads = {
    'progress_optimization.gif': all_urls['progress_gif'],
    'final_comparison.png': all_urls['final_comparison'], 
    'what_displays_show.png': all_urls['what_displays_show'],
    'what_eye_sees.png': all_urls['what_eye_sees'],
    'focal_length_sweep.gif': all_urls['focal_length_sweep'],
    'eye_movement_sweep.gif': all_urls['eye_movement_sweep'],
    'loss_history.json': all_urls['loss_history_json'],
    'focal_data.json': all_urls['focal_data_json']
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

# Download all files
print("="*60)
downloaded_count = 0

for filename, url in downloads.items():
    local_path = os.path.join(results_dir, filename)
    success = download_file(url, local_path)
    if success:
        downloaded_count += 1

# Save optimization metadata
metadata = {
    'optimization_timestamp': output['timestamp'],
    'final_loss': output['final_loss'],
    'iterations': output['optimization_specs']['iterations'],
    'resolution': output['optimization_specs']['resolution'],
    'rays_per_pixel': output['optimization_specs']['rays_per_pixel'],
    'display_resolution': output['optimization_specs']['display_resolution'],
    'focal_planes': output['optimization_specs']['focal_planes'],
    'display_focal_lengths_mm': output['display_focal_lengths_mm'],
    'frames_counts': {
        'progress_gif': output['optimization_specs']['frames_in_progress_gif'],
        'focal_sweep_gif': output['optimization_specs']['frames_in_focal_sweep'],
        'eye_movement_gif': output['optimization_specs']['frames_in_eye_movement']
    },
    'download_urls': all_urls,
    'execution_stats': {
        'delay_time_seconds': response_data['delayTime'],
        'execution_time_seconds': response_data['executionTime'],
        'worker_id': response_data['workerId']
    }
}

metadata_path = os.path.join(results_dir, 'optimization_metadata.json')
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print("="*60)
print(f"📊 DOWNLOAD SUMMARY:")
print(f"✅ Successfully downloaded: {downloaded_count}/{len(downloads)} files")
print(f"📁 All files saved to: {results_dir}/")
print(f"📋 Metadata saved to: {metadata_path}")

# Display directory structure
print(f"\n📁 RESULTS DIRECTORY:")
for file in sorted(os.listdir(results_dir)):
    file_path = os.path.join(results_dir, file)
    if os.path.isfile(file_path):
        file_size = os.path.getsize(file_path) / 1024**2
        print(f"   {file} ({file_size:.1f} MB)")

print(f"\n📊 OPTIMIZATION RESULTS:")
print(f"   Final Loss: {output['final_loss']:.6f}")
print(f"   Iterations: {output['optimization_specs']['iterations']}")
print(f"   Display Focal Lengths: {output['display_focal_lengths_mm']} mm")
print(f"   Execution Time: {response_data['executionTime']/1000:.1f} seconds")

print(f"\n✅ All optimization results downloaded and organized!")