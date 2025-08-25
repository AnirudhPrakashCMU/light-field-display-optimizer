#!/usr/bin/env python3
"""
Manage RunPod Workers and Run Enhanced Optimization
Delete wasteful workers and run memory-optimized version
"""

import requests
import json
import time
import os

API_KEY = os.getenv("RUNPOD_API_KEY", "rpa_LDVD1JMIAKGTVVO4AAV82Y5PN2FJTC2ROG5N6K6N1a1hn7")
ENDPOINT_ID = "d93rynzpivo6va"

def delete_all_workers():
    """Delete all current workers to reset"""
    
    print("🗑️  Deleting all workers...")
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Get current workers
    try:
        health_response = requests.get(
            f"https://api.runpod.ai/v2/{ENDPOINT_ID}/health",
            headers=headers,
            timeout=15
        )
        
        if health_response.status_code == 200:
            health = health_response.json()
            workers = health.get('workers', {})
            
            total_workers = sum([
                workers.get('ready', 0),
                workers.get('idle', 0), 
                workers.get('running', 0),
                workers.get('initializing', 0)
            ])
            
            print(f"📊 Current workers: {total_workers} total")
            
            if total_workers == 0:
                print(f"✅ No workers to delete")
                return True
            
            # Scale down to 0 workers
            scale_payload = {
                "workersMin": 0,
                "workersMax": 0
            }
            
            scale_response = requests.patch(
                f"https://api.runpod.ai/v2/{ENDPOINT_ID}",
                headers=headers,
                json=scale_payload,
                timeout=30
            )
            
            if scale_response.status_code == 200:
                print(f"✅ Scaled down to 0 workers")
                
                # Wait for workers to terminate
                print(f"⏳ Waiting for workers to terminate...")
                for i in range(20):  # Wait up to 10 minutes
                    time.sleep(30)
                    
                    check_response = requests.get(
                        f"https://api.runpod.ai/v2/{ENDPOINT_ID}/health",
                        headers=headers,
                        timeout=15
                    )
                    
                    if check_response.status_code == 200:
                        check_health = check_response.json()
                        check_workers = check_health.get('workers', {})
                        
                        remaining = sum([
                            check_workers.get('ready', 0),
                            check_workers.get('idle', 0),
                            check_workers.get('running', 0),
                            check_workers.get('initializing', 0)
                        ])
                        
                        print(f"   [{i*30}s] Workers remaining: {remaining}")
                        
                        if remaining == 0:
                            print(f"✅ All workers deleted")
                            return True
                
                print(f"⏰ Timeout waiting for worker deletion")
                return False
            else:
                print(f"❌ Scale down failed: {scale_response.status_code}")
                return False
        else:
            print(f"❌ Health check failed: {health_response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Worker deletion error: {e}")
        return False

def scale_up_optimized():
    """Scale up with optimized worker configuration"""
    
    print(f"\n🚀 Scaling up with optimized configuration...")
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Scale up to 1 worker for testing
    scale_payload = {
        "workersMin": 1,
        "workersMax": 1
    }
    
    try:
        response = requests.patch(
            f"https://api.runpod.ai/v2/{ENDPOINT_ID}",
            headers=headers,
            json=scale_payload,
            timeout=30
        )
        
        if response.status_code == 200:
            print(f"✅ Scaled up to 1 optimized worker")
            
            # Wait for worker to be ready
            print(f"⏳ Waiting for worker to initialize...")
            for i in range(20):  # Wait up to 10 minutes
                time.sleep(30)
                
                health_response = requests.get(
                    f"https://api.runpod.ai/v2/{ENDPOINT_ID}/health",
                    headers=headers,
                    timeout=15
                )
                
                if health_response.status_code == 200:
                    health = health_response.json()
                    workers = health.get('workers', {})
                    
                    ready = workers.get('ready', 0)
                    idle = workers.get('idle', 0)
                    
                    print(f"   [{i*30}s] Ready: {ready}, Idle: {idle}")
                    
                    if ready > 0 or idle > 0:
                        print(f"✅ Worker ready!")
                        return True
            
            print(f"⏰ Timeout waiting for worker")
            return False
        else:
            print(f"❌ Scale up failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Scale up error: {e}")
        return False

def run_memory_optimized_test():
    """Run optimization that actually uses the available memory"""
    
    print(f"\n🧠 Running MEMORY-OPTIMIZED light field optimization...")
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Enhanced payload to use more memory
    payload = {
        "input": {
            "task_type": "full_optimization",
            "iterations": 200,           # More iterations
            "resolution": 1024,          # Much higher resolution  
            "rays_per_pixel": 32,        # More rays per pixel
            "scenes": ["spherical_checkerboard"],
            "save_large_outputs": True,
            "batch_size": 8192,          # Larger batches
            "target_memory_gb": 20       # Target 20GB usage
        }
    }
    
    print(f"📤 Enhanced optimization parameters:")
    print(f"   Iterations: 200")
    print(f"   Resolution: 1024x1024") 
    print(f"   Rays per pixel: 32")
    print(f"   Target memory: 20GB")
    print(f"   Expected runtime: ~10-15 minutes")
    
    try:
        start_time = time.time()
        response = requests.post(
            f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync",
            headers=headers,
            json=payload,
            timeout=1800  # 30 minute timeout
        )
        
        elapsed = time.time() - start_time
        
        print(f"⏱️  Optimization completed in {elapsed/60:.1f} minutes")
        print(f"📊 Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            if result.get('status') == 'COMPLETED':
                output = result.get('output', {})
                
                if output.get('status') == 'success':
                    print(f"\n🎉 MEMORY-OPTIMIZED OPTIMIZATION SUCCESSFUL!")
                    
                    results = output.get('results', {})
                    gpu_info = output.get('gpu_info', {})
                    
                    print(f"🎯 Enhanced Results:")
                    print(f"   Scenes: {results.get('scenes_completed', [])}")
                    print(f"   Peak GPU memory: {results.get('gpu_memory_peak', 0):.2f} GB")
                    print(f"   GPU: {gpu_info.get('gpu_name', 'N/A')}")
                    print(f"   Total GPU memory: {gpu_info.get('gpu_memory_total', 0):.1f} GB")
                    
                    # Show scene results
                    scene_results = results.get('results', {})
                    for scene, data in scene_results.items():
                        if isinstance(data, dict):
                            final_loss = data.get('final_loss', 'N/A')
                            iterations = data.get('iterations', 'N/A')
                            memory_used = data.get('gpu_memory_used', 0)
                            print(f"   {scene}: Loss = {final_loss}, Memory = {memory_used:.2f} GB")
                    
                    return True
                else:
                    print(f"❌ Optimization failed: {output.get('message')}")
                    return False
            else:
                print(f"❌ Job failed: {result}")
                return False
        else:
            print(f"❌ Request failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Main worker management and optimization"""
    
    print("🎯 RUNPOD WORKER MANAGEMENT & OPTIMIZATION")
    print("="*50)
    
    # Step 1: Delete all workers
    deleted = delete_all_workers()
    
    if deleted:
        # Step 2: Scale up optimized
        scaled = scale_up_optimized()
        
        if scaled:
            # Step 3: Run memory-optimized test
            success = run_memory_optimized_test()
            
            if success:
                print(f"\n🎉 SUCCESS: Memory-optimized light field optimizer working!")
                print(f"🚀 Endpoint ready for production use")
                return True
            else:
                print(f"\n❌ Optimization failed")
                return False
        else:
            print(f"\n❌ Could not scale up workers")
            return False
    else:
        print(f"\n❌ Could not delete workers")
        return False

if __name__ == "__main__":
    main()