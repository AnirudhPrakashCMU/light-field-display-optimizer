#!/usr/bin/env python3
"""
Debug RunPod Endpoint - Simple Test
Test the endpoint with minimal payloads to identify issues
"""

import requests
import json
import time

API_KEY = "rpa_LDVD1JMIAKGTVVO4AAV82Y5PN2FJTC2ROG5N6K6N1a1hn7"
ENDPOINT_ID = "d93rynzpivo6va"

def test_minimal_payload():
    """Test with minimal payload"""
    
    print("🔧 Testing minimal payload...")
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Minimal test payload
    payload = {
        "input": {
            "test": "hello"
        }
    }
    
    try:
        response = requests.post(
            f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync",
            headers=headers,
            json=payload,
            timeout=120
        )
        
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Minimal test successful!")
            return True
        else:
            print(f"❌ Minimal test failed")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_async_and_poll():
    """Test async submission and polling"""
    
    print(f"\n🔄 Testing async submission...")
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "input": {
            "task_type": "quick_test",
            "iterations": 10,
            "resolution": 128
        }
    }
    
    try:
        # Submit async
        response = requests.post(
            f"https://api.runpod.ai/v2/{ENDPOINT_ID}/run",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        print(f"Async submit status: {response.status_code}")
        print(f"Async response: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            job_id = result.get('id')
            
            if job_id:
                print(f"✅ Job submitted: {job_id}")
                
                # Poll for results
                print(f"📊 Polling for results...")
                
                for i in range(20):  # Poll for 10 minutes
                    time.sleep(30)
                    
                    status_response = requests.get(
                        f"https://api.runpod.ai/v2/{ENDPOINT_ID}/status/{job_id}",
                        headers=headers,
                        timeout=15
                    )
                    
                    if status_response.status_code == 200:
                        status_data = status_response.json()
                        status = status_data.get('status', 'UNKNOWN')
                        
                        print(f"   [{i*30}s] Status: {status}")
                        
                        if status == 'COMPLETED':
                            output = status_data.get('output', {})
                            print(f"✅ Async job completed!")
                            print(f"Output: {json.dumps(output, indent=2)}")
                            return True
                        elif status == 'FAILED':
                            error = status_data.get('error', 'Unknown error')
                            print(f"❌ Async job failed: {error}")
                            return False
                    else:
                        print(f"   Status check failed: {status_response.status_code}")
                
                print(f"⏰ Async job timeout")
                return False
            else:
                print(f"❌ No job ID returned")
                return False
        else:
            print(f"❌ Async submit failed")
            return False
            
    except Exception as e:
        print(f"❌ Async error: {e}")
        return False

def check_endpoint_health():
    """Check if endpoint is healthy and ready"""
    
    print("🏥 Checking endpoint health...")
    
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
        
        print(f"Health check status: {response.status_code}")
        print(f"Health response: {response.text}")
        
        return response.status_code == 200
        
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def main():
    """Main debugging function"""
    
    print("🔧 DEBUGGING RUNPOD ENDPOINT")
    print(f"Endpoint: {ENDPOINT_ID}")
    print("="*40)
    
    # Step 1: Health check
    healthy = check_endpoint_health()
    
    # Step 2: Minimal test
    minimal_works = test_minimal_payload()
    
    # Step 3: Async test (if minimal works)
    if minimal_works:
        async_works = test_async_and_poll()
        
        if async_works:
            print(f"\n🎉 ENDPOINT IS FULLY FUNCTIONAL!")
            return True
    
    print(f"\n🔧 DEBUGGING SUMMARY:")
    print(f"   Health check: {'✅' if healthy else '❌'}")
    print(f"   Minimal test: {'✅' if minimal_works else '❌'}")
    
    if not minimal_works:
        print(f"\n🚨 ISSUES FOUND:")
        print(f"   • Endpoint may not be properly initialized")
        print(f"   • Handler code may have errors")
        print(f"   • Docker build may have failed")
        print(f"\n🔧 CHECK:")
        print(f"   • RunPod dashboard logs")
        print(f"   • Endpoint build status")
        print(f"   • Handler code syntax")
    
    return False

if __name__ == "__main__":
    main()