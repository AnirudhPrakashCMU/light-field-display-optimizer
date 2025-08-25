#!/usr/bin/env python3
"""
Minimal Endpoint Test
Test with absolute minimal payload to see if handler responds at all
"""

import requests
import json
import os

API_KEY = os.getenv("RUNPOD_API_KEY", "rpa_LDVD1JMIAKGTVVO4AAV82Y5PN2FJTC2ROG5N6K6N1a1hn7")
ENDPOINT_ID = "d93rynzpivo6va"

def test_empty_input():
    """Test with completely empty input"""
    
    print("🧪 Testing with empty input...")
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {"input": {}}
    
    try:
        response = requests.post(
            f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync",
            headers=headers,
            json=payload,
            timeout=120
        )
        
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")
        
        # Check if we get actual output (not just IN_QUEUE)
        if result.get('status') != 'IN_QUEUE':
            output = result.get('output', {})
            if output:
                print(f"✅ Handler is responding!")
                return True
        
        print(f"❌ Still stuck in queue")
        return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_hello_world():
    """Test with simple hello world"""
    
    print(f"\n👋 Testing hello world...")
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "input": {
            "message": "hello world"
        }
    }
    
    try:
        response = requests.post(
            f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync",
            headers=headers,
            json=payload,
            timeout=60
        )
        
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")
        
        return response.status_code == 200 and result.get('status') != 'IN_QUEUE'
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🔧 MINIMAL ENDPOINT TESTING")
    print("Testing if handler responds to any input at all")
    print("="*45)
    
    # Test 1: Empty input
    empty_works = test_empty_input()
    
    # Test 2: Simple input  
    hello_works = test_hello_world()
    
    if empty_works or hello_works:
        print(f"\n✅ Handler is working!")
        print(f"🎯 Try running the optimizer now")
    else:
        print(f"\n❌ Handler not responding")
        print(f"🔧 Check RunPod dashboard for:")
        print(f"   • Python import errors")
        print(f"   • Handler startup crashes")
        print(f"   • CUDA initialization issues")
        print(f"\n🎯 The endpoint needs the improved handler code!")
        print(f"   Current version may have import/runtime errors")