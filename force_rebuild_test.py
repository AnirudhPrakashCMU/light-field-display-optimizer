#!/usr/bin/env python3
"""
Force Rebuild Test
Check if endpoint has rebuilt and test with different approaches
"""

import requests
import json
import time
import os

API_KEY = os.getenv("RUNPOD_API_KEY", "rpa_LDVD1JMIAKGTVVO4AAV82Y5PN2FJTC2ROG5N6K6N1a1hn7")
ENDPOINT_ID = "d93rynzpivo6va"

def check_workers_version():
    """Check if workers have updated to new version"""
    
    print("🔍 Checking worker versions...")
    
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
            health = response.json()
            workers = health.get('workers', {})
            
            print(f"👷 Current workers:")
            print(f"   Ready: {workers.get('ready', 0)}")
            print(f"   Running: {workers.get('running', 0)}")
            print(f"   Initializing: {workers.get('initializing', 0)}")
            print(f"   Idle: {workers.get('idle', 0)}")
            
            # If workers are initializing, they might be rebuilding
            if workers.get('initializing', 0) > 0:
                print(f"⏳ Workers initializing - may be rebuilding with new code")
                return False
            elif workers.get('ready', 0) > 0 or workers.get('idle', 0) > 0:
                print(f"✅ Workers ready - testing if they use new handler")
                return True
            else:
                print(f"❌ No ready workers")
                return False
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_with_explicit_logging():
    """Test with a payload that should generate logs"""
    
    print(f"\n📋 Testing with explicit logging request...")
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "input": {
            "debug": True,
            "test_gpu": True,
            "simple_task": True
        }
    }
    
    try:
        print(f"📤 Sending debug request...")
        response = requests.post(
            f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync",
            headers=headers,
            json=payload,
            timeout=180  # 3 minute timeout
        )
        
        print(f"📊 Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"📋 Response: {json.dumps(result, indent=2)}")
            
            # Check if it's still in queue
            if result.get('status') == 'IN_QUEUE':
                print(f"⚠️  Still in queue - handler likely crashing on startup")
                return False
            else:
                output = result.get('output', {})
                if output and output.get('status') != 'error':
                    print(f"✅ Handler executed successfully!")
                    return True
                else:
                    print(f"❌ Handler executed but returned error: {output}")
                    return False
        else:
            print(f"❌ Request failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Main rebuild testing"""
    
    print("🔄 FORCE REBUILD TESTING")
    print(f"Endpoint: {ENDPOINT_ID}")
    print("="*35)
    
    # Check worker status
    workers_ready = check_workers_version()
    
    if workers_ready:
        # Test the improved handler
        handler_works = test_with_explicit_logging()
        
        if handler_works:
            print(f"\n🎉 SUCCESS: Improved handler is working!")
            print(f"🚀 Now ready to run light field optimization")
            
            # Run actual optimization test
            print(f"\n🧪 Testing actual optimization...")
            
            opt_payload = {
                "input": {
                    "task_type": "quick_test",
                    "iterations": 10,
                    "resolution": 128
                }
            }
            
            headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
            
            try:
                opt_response = requests.post(
                    f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync",
                    headers=headers,
                    json=opt_payload,
                    timeout=300
                )
                
                if opt_response.status_code == 200:
                    opt_result = opt_response.json()
                    output = opt_result.get('output', {})
                    
                    if output.get('status') == 'success':
                        print(f"🎉 LIGHT FIELD OPTIMIZATION WORKING!")
                        results = output.get('results', {})
                        print(f"   Final loss: {results.get('final_loss', 'N/A')}")
                        print(f"   GPU memory: {results.get('gpu_memory_used', 'N/A'):.2f} GB")
                        return True
                    else:
                        print(f"❌ Optimization failed: {output.get('message')}")
                        return False
                else:
                    print(f"❌ Optimization request failed: {opt_response.status_code}")
                    return False
                    
            except Exception as e:
                print(f"❌ Optimization test error: {e}")
                return False
        else:
            print(f"\n❌ Handler still not working after rebuild")
    else:
        print(f"\n⏳ Workers not ready yet - wait for rebuild to complete")
    
    print(f"\n🔧 NEXT STEPS:")
    print(f"1. Check RunPod dashboard for endpoint rebuild status")
    print(f"2. Look at worker logs for specific Python errors")  
    print(f"3. Wait 2-3 minutes for complete rebuild")
    print(f"4. Try this test again")
    
    return False

if __name__ == "__main__":
    main()