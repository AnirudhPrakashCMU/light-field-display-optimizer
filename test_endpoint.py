#!/usr/bin/env python3
"""
Test RunPod Serverless Endpoint - Light Field Optimizer
Test the deployed endpoint with various optimization configurations
Endpoint ID: d93rynzpivo6va
"""

import requests
import json
import time
import os
from datetime import datetime

# RunPod endpoint details from the UI
API_KEY = os.getenv("RUNPOD_API_KEY", "your_api_key_here")
ENDPOINT_ID = "d93rynzpivo6va"
ENDPOINT_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"

def test_quick_optimization():
    """Test quick checkerboard optimization"""
    
    print("🧪 Testing Quick Checkerboard Optimization...")
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "input": {
            "task_type": "quick_test",
            "iterations": 25,
            "resolution": 256,
            "rays_per_pixel": 8,
            "job_id": f"quick_test_{datetime.now().strftime('%H%M%S')}"
        }
    }
    
    print(f"📤 Sending request: {json.dumps(payload, indent=2)}")
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{ENDPOINT_URL}/runsync",
            headers=headers,
            json=payload,
            timeout=600  # 10 minute timeout
        )
        
        elapsed = time.time() - start_time
        
        print(f"⏱️  Request completed in {elapsed:.1f} seconds")
        print(f"📊 Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"📋 Raw response:")
            print(json.dumps(result, indent=2))
            
            if 'error' in result:
                print(f"❌ Endpoint error: {result['error']}")
                return False
            
            output = result.get('output', {})
            if output.get('status') == 'success':
                print(f"\n✅ QUICK TEST SUCCESSFUL!")
                
                results = output.get('results', {})
                gpu_info = output.get('gpu_info', {})
                
                print(f"🎯 Optimization Results:")
                print(f"   Final loss: {results.get('final_loss', 'N/A')}")
                print(f"   Iterations: {results.get('iterations', 'N/A')}")
                print(f"   Resolution: {results.get('resolution', 'N/A')}x{results.get('resolution', 'N/A')}")
                print(f"   Rays per pixel: {results.get('rays_per_pixel', 'N/A')}")
                print(f"   GPU memory used: {results.get('gpu_memory_used', 'N/A'):.2f} GB")
                
                print(f"\n🖥️  GPU Information:")
                print(f"   GPU: {gpu_info.get('gpu_name', 'N/A')}")
                print(f"   Total memory: {gpu_info.get('gpu_memory_total', 'N/A'):.1f} GB")
                print(f"   Available: {gpu_info.get('gpu_memory_available', 'N/A'):.1f} GB")
                
                return True
            else:
                print(f"❌ Optimization failed: {output.get('message', 'Unknown error')}")
                return False
                
        else:
            print(f"❌ Request failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request error: {e}")
        return False

def test_full_optimization():
    """Test full optimization with spherical checkerboard"""
    
    print(f"\n🚀 Testing Full Optimization...")
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "input": {
            "task_type": "full_optimization",
            "iterations": 100,
            "resolution": 384,
            "scenes": ["spherical_checkerboard"],
            "save_large_outputs": True,
            "job_id": f"full_test_{datetime.now().strftime('%H%M%S')}"
        }
    }
    
    print(f"📤 Sending full optimization request...")
    print(f"⏱️  Expected runtime: ~5-10 minutes")
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{ENDPOINT_URL}/runsync",
            headers=headers,
            json=payload,
            timeout=1200  # 20 minute timeout
        )
        
        elapsed = time.time() - start_time
        
        print(f"⏱️  Full optimization completed in {elapsed/60:.1f} minutes")
        print(f"📊 Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            if 'error' in result:
                print(f"❌ Endpoint error: {result['error']}")
                return False
            
            output = result.get('output', {})
            if output.get('status') == 'success':
                print(f"\n🎉 FULL OPTIMIZATION SUCCESSFUL!")
                
                results = output.get('results', {})
                
                print(f"🎯 Full Results:")
                print(f"   Scenes completed: {results.get('scenes_completed', [])}")
                print(f"   Total scenes: {results.get('total_scenes', 'N/A')}")
                print(f"   Peak GPU memory: {results.get('gpu_memory_peak', 'N/A'):.2f} GB")
                
                # Check individual scene results
                scene_results = results.get('results', {})
                for scene, scene_data in scene_results.items():
                    if isinstance(scene_data, dict) and 'final_loss' in scene_data:
                        print(f"   {scene}: Loss = {scene_data['final_loss']:.6f}")
                
                return True
            else:
                print(f"❌ Full optimization failed: {output.get('message', 'Unknown error')}")
                return False
                
        else:
            print(f"❌ Request failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request error: {e}")
        return False

def test_async_optimization():
    """Test asynchronous optimization (non-blocking)"""
    
    print(f"\n🔄 Testing Async Optimization...")
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "input": {
            "task_type": "quick_test",
            "iterations": 50,
            "resolution": 320,
            "rays_per_pixel": 12,
            "job_id": f"async_test_{datetime.now().strftime('%H%M%S')}"
        }
    }
    
    try:
        # Submit async job
        response = requests.post(
            f"{ENDPOINT_URL}/run",  # Use /run for async
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            job_id = result.get('id')
            
            print(f"✅ Async job submitted: {job_id}")
            
            # Poll for completion
            print(f"📊 Monitoring async job...")
            
            for i in range(60):  # Poll for up to 30 minutes
                status_response = requests.get(
                    f"{ENDPOINT_URL}/status/{job_id}",
                    headers=headers,
                    timeout=15
                )
                
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    status = status_data.get('status', 'UNKNOWN')
                    
                    elapsed = i * 30
                    print(f"   [{elapsed}s] Status: {status}")
                    
                    if status == 'COMPLETED':
                        output = status_data.get('output', {})
                        print(f"✅ Async optimization completed!")
                        if output.get('status') == 'success':
                            results = output.get('results', {})
                            print(f"   Final loss: {results.get('final_loss', 'N/A')}")
                            return True
                        else:
                            print(f"❌ Optimization failed: {output.get('message')}")
                            return False
                    elif status == 'FAILED':
                        print(f"❌ Async job failed")
                        return False
                
                time.sleep(30)  # Wait 30 seconds
            
            print(f"⏰ Async job timeout")
            return False
            
        else:
            print(f"❌ Async submission failed: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Async test error: {e}")
        return False

def main():
    """Main endpoint testing function"""
    
    print("🎯 TESTING RUNPOD SERVERLESS ENDPOINT")
    print(f"Endpoint ID: {ENDPOINT_ID}")
    print(f"Repository: AnirudhPrakashCMU/light-field-display-optimizer")
    print(f"GPU Type: A100 80GB")
    print("="*60)
    
    # Test 1: Quick synchronous test
    quick_success = test_quick_optimization()
    
    if quick_success:
        print(f"\n✅ Endpoint is working! Testing more configurations...")
        
        # Test 2: Full optimization
        full_success = test_full_optimization()
        
        if full_success:
            print(f"\n🎉 ALL TESTS SUCCESSFUL!")
            print(f"🚀 Light Field Optimizer endpoint is fully operational!")
            
            # Test 3: Async capability
            async_success = test_async_optimization()
            
            return True
    
    print(f"\n❌ Endpoint testing failed")
    print(f"🔧 Check endpoint logs in RunPod dashboard")
    print(f"🔗 Endpoint URL: {ENDPOINT_URL}")
    
    return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print(f"\n🎯 ENDPOINT READY FOR PRODUCTION!")
        print(f"📡 API: {ENDPOINT_URL}")
        print(f"🔑 Auth: Bearer {API_KEY}")
        print(f"\n📝 Usage Examples:")
        print(f'   Quick: {{"input": {{"task_type": "quick_test", "iterations": 50}}}}')
        print(f'   Full:  {{"input": {{"task_type": "full_optimization", "iterations": 200}}}}')
    else:
        print(f"\n🔧 Troubleshooting needed - check RunPod logs")