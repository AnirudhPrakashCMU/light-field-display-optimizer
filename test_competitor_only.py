#!/usr/bin/env python3
"""
Quick test of competitor inverse rendering system only
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

# Import from main file
from standalone_optimizer import (
    device, create_scene_objects, competitor_inverse_rendering, 
    generate_competitor_outputs
)

def main():
    print("🏁 Testing Competitor Inverse Rendering System Only")
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f'/workspace/competitor_test_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    print(f"📁 Results directory: {results_dir}")
    
    resolution = 128
    
    # Test all scenes
    scene_names = ['basic', 'complex', 'stick_figure', 'layered', 'office', 'nature', 'textured_basic']
    
    for scene_name in scene_names:
        print(f"\n🏁 COMPETITOR: {scene_name}")
        
        scene_objects = create_scene_objects(scene_name)
        
        if isinstance(scene_objects, list):
            # Run competitor system
            competitor_display = competitor_inverse_rendering(scene_objects, resolution)
            
            # Generate debug outputs
            competitor_result = generate_competitor_outputs(
                scene_name, competitor_display, scene_objects, resolution, results_dir
            )
            
            print(f"✅ {scene_name} competitor complete")
            print(f"   Displays: {competitor_result.get('displays_url', 'Local only')}")
        else:
            print(f"⏭️  Skipping {scene_name} (SphericalCheckerboard)")
    
    print(f"\n🎉 All competitor tests complete!")
    print(f"📁 Results saved to: {results_dir}")

if __name__ == "__main__":
    main()