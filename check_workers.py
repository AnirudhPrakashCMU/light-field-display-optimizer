#!/usr/bin/env python3
"""
Check RunPod Endpoint Worker Status
Monitor worker health and try to identify startup issues
"""

import requests
import json
import time

API_KEY = "rpa_LDVD1JMIAKGTVVO4AAV82Y5PN2FJTC2ROG5N6K6N1a1hn7"
ENDPOINT_ID = "d93rynzpivo6va"

def check_worker_status():
    """Check detailed worker status"""
    
    print("🔍 Checking worker status...")
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(
            f"https://api.runpod.ai/v2/{ENDPOINT_ID}/health",
            headers=headers,
            timeout=15
        )
        
        if response.status_code == 200:
            health_data = response.json()
            print(f"📊 Health data: {json.dumps(health_data, indent=2)}")
            
            workers = health_data.get('workers', {})
            jobs = health_data.get('jobs', {})
            
            print(f"\n👷 Worker Status:")
            print(f"   Ready: {workers.get('ready', 0)}")
            print(f"   Initializing: {workers.get('initializing', 0)}")
            print(f"   Running: {workers.get('running', 0)}")
            print(f"   Idle: {workers.get('idle', 0)}")
            print(f"   Unhealthy: {workers.get('unhealthy', 0)}")
            
            print(f"\n📋 Job Queue:")
            print(f"   In Queue: {jobs.get('inQueue', 0)}")
            print(f"   In Progress: {jobs.get('inProgress', 0)}")
            print(f"   Completed: {jobs.get('completed', 0)}")
            print(f"   Failed: {jobs.get('failed', 0)}")
            
            # Check if workers are ready
            if workers.get('ready', 0) > 0:
                print(f"✅ Workers are ready!")
                return True
            elif workers.get('initializing', 0) > 0:
                print(f"⏳ Workers still initializing...")
                return False
            elif workers.get('unhealthy', 0) > 0:
                print(f"❌ Workers are unhealthy!")
                return False
            else:
                print(f"⚠️  No workers detected")
                return False
                
        else:
            print(f"❌ Health check failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def wait_for_workers_ready(max_wait_minutes=10):
    """Wait for workers to become ready"""
    
    print(f"⏳ Waiting for workers to become ready (max {max_wait_minutes} minutes)...")
    
    start_time = time.time()
    
    while True:
        ready = check_worker_status()
        
        if ready:
            print(f"✅ Workers are ready!")
            return True
        
        elapsed = (time.time() - start_time) / 60
        if elapsed >= max_wait_minutes:
            print(f"⏰ Timeout after {max_wait_minutes} minutes")
            return False
        
        print(f"   Waiting... ({elapsed:.1f}m elapsed)")
        time.sleep(30)

def test_simple_optimization():
    """Test simple optimization once workers are ready"""
    
    print(f"\n🚀 Testing simple optimization...")
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "input": {
            "task_type": "quick_test",
            "iterations": 5,  # Very small test
            "resolution": 128,
            "rays_per_pixel": 4
        }
    }
    
    try:
        response = requests.post(
            f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync",
            headers=headers,
            json=payload,
            timeout=300  # 5 minute timeout
        )
        
        print(f"Optimization status: {response.status_code}")
        result = response.json()
        print(f"Result: {json.dumps(result, indent=2)}")
        
        if response.status_code == 200 and result.get('status') != 'IN_QUEUE':
            print(f"✅ Optimization executed!")
            
            output = result.get('output', {})
            if output.get('status') == 'success':
                print(f"🎉 LIGHT FIELD OPTIMIZATION SUCCESSFUL!")
                results = output.get('results', {})
                print(f"   Final loss: {results.get('final_loss', 'N/A')}")
                print(f"   GPU memory: {results.get('gpu_memory_used', 'N/A'):.2f} GB")
                return True
            else:
                print(f"❌ Optimization failed: {output.get('message', 'Unknown')}")
                return False
        else:
            print(f"❌ Still stuck in queue or failed")
            return False
            
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def main():
    """Main worker monitoring and testing"""
    
    print("🎯 RUNPOD ENDPOINT WORKER MONITORING")
    print("="*40)
    
    # Step 1: Check current status
    print("📊 Current status:")
    check_worker_status()
    
    # Step 2: Wait for workers to be ready
    workers_ready = wait_for_workers_ready(max_wait_minutes=15)
    
    if workers_ready:
        # Step 3: Test optimization
        optimization_works = test_simple_optimization()
        
        if optimization_works:
            print(f"\n🎉 SUCCESS: Light field optimizer is working!")
            print(f"🚀 Ready for full optimization runs")
            return True
        else:
            print(f"\n❌ Workers ready but optimization failed")
            return False
    else:
        print(f"\n❌ Workers never became ready")
        print(f"🔧 Check RunPod dashboard for worker logs")
        return False

if __name__ == "__main__":
    main()