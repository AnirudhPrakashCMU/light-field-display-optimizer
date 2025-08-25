#!/usr/bin/env python3
"""
Test Handler Directly
Test if the rp_handler.py is working with minimal inputs
"""

import requests
import json
import time

API_KEY = "rpa_LDVD1JMIAKGTVVO4AAV82Y5PN2FJTC2ROG5N6K6N1a1hn7"
ENDPOINT_ID = "d93rynzpivo6va"

def test_minimal_handler():
    """Test handler with absolute minimal input"""
    
    print("🧪 Testing minimal handler input...")
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Test just the handler without any optimization
    payload = {
        "input": {}  # Empty input to test basic handler
    }
    
    try:
        print("📤 Sending empty input test...")
        response = requests.post(
            f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync",
            headers=headers,
            json=payload,
            timeout=60
        )
        
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Result: {json.dumps(result, indent=2)}")
        
        return response.status_code == 200 and result.get('status') != 'IN_QUEUE'
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_basic_task():
    """Test with basic task type"""
    
    print(f"\n🔧 Testing basic task...")
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "input": {
            "task_type": "quick_test",
            "iterations": 1  # Minimal iterations
        }
    }
    
    try:
        print("📤 Sending basic task...")
        response = requests.post(
            f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync",
            headers=headers,
            json=payload,
            timeout=180  # 3 minute timeout
        )
        
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Result: {json.dumps(result, indent=2)}")
        
        if response.status_code == 200:
            if result.get('status') == 'IN_QUEUE':
                print(f"⚠️  Still in queue - handler may be crashing")
                return False
            else:
                output = result.get('output', {})
                if output:
                    print(f"✅ Handler executed!")
                    print(f"   Status: {output.get('status', 'N/A')}")
                    if 'message' in output:
                        print(f"   Message: {output['message']}")
                    return True
                else:
                    print(f"❌ No output from handler")
                    return False
        else:
            print(f"❌ Request failed")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def check_logs():
    """Check endpoint logs if available"""
    
    print(f"\n📋 Checking logs...")
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        # Try to get logs
        response = requests.get(
            f"https://api.runpod.ai/v2/{ENDPOINT_ID}/logs",
            headers=headers,
            timeout=15
        )
        
        print(f"Logs status: {response.status_code}")
        
        if response.status_code == 200:
            logs = response.json()
            print(f"📋 Logs: {json.dumps(logs, indent=2)}")
        else:
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Logs error: {e}")

def main():
    """Main testing sequence"""
    
    print("🔧 HANDLER DIRECT TESTING")
    print("="*30)
    
    # Wait a moment for workers to be ready
    time.sleep(10)
    
    # Test 1: Minimal handler
    minimal_works = test_minimal_handler()
    
    if minimal_works:
        print(f"✅ Handler responding!")
        
        # Test 2: Basic optimization
        basic_works = test_basic_task()
        
        if basic_works:
            print(f"\n🎉 OPTIMIZATION IS WORKING!")
            print(f"🚀 Ready for full runs")
            return True
        else:
            print(f"\n❌ Handler works but optimization fails")
    else:
        print(f"❌ Handler not responding properly")
    
    # Check logs for debugging
    check_logs()
    
    print(f"\n🔧 RECOMMENDATION:")
    print(f"   Check RunPod dashboard logs for:")
    print(f"   • Python import errors")
    print(f"   • CUDA initialization issues") 
    print(f"   • Handler crash logs")
    
    return False

if __name__ == "__main__":
    main()