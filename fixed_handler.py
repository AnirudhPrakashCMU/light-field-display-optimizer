"""
Fixed RunPod Handler - Copy this content to replace rp_handler.py
Minimal version to avoid import issues
"""

import runpod
import torch
import numpy as np
import json
from datetime import datetime

def handler(job):
    """Simple working handler"""
    try:
        print(f"🚀 Handler started")
        
        inp = job.get("input", {})
        task_type = inp.get("task_type", "test")
        
        # Simple GPU test
        if torch.cuda.is_available():
            device = torch.device("cuda")
            test_tensor = torch.randn(100, 100, device=device)
            gpu_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        else:
            gpu_name = "No GPU"
            memory_gb = 0
        
        result = {
            'status': 'success',
            'message': f'Handler working! Task: {task_type}',
            'gpu_name': gpu_name,
            'gpu_memory_gb': memory_gb,
            'input_received': inp,
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"✅ Returning: {result}")
        return result
        
    except Exception as e:
        return {
            'status': 'error', 
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }

runpod.serverless.start({"handler": handler})