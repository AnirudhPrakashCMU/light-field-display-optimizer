#!/usr/bin/env python3
"""
Simple RunPod Endpoint Test
Test if we can invoke an existing endpoint or create a simple one
"""

import requests
import json
import time

API_KEY = "rpa_LDVD1JMIAKGTVVO4AAV82Y5PN2FJTC2ROG5N6K6N1a1hn7"

def test_simple_endpoint_creation():
    """Try the simplest possible endpoint creation"""
    
    print("🧪 Testing simple endpoint creation...")
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Use a public template if available
    payload = {
        "name": f"light-field-test-{int(time.time())}",
        "templateId": "runpod-workers-gpu",  # Generic GPU template
        "gpuTypeIds": ["NVIDIA A100 80GB PCIe"],
        "scalerType": "QUEUE_DELAY",
        "scalerValue": 4,
        "workersMin": 0,
        "workersMax": 1,
        "idleTimeout": 5,
        "executionTimeoutMs": 600000
    }
    
    try:
        # Try REST API instead of GraphQL
        response = requests.post(
            "https://api.runpod.io/v1/endpoints",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code in [200, 201]:
            data = response.json()
            endpoint_id = data.get('id')
            if endpoint_id:
                print(f"✅ Endpoint created: {endpoint_id}")
                return endpoint_id
        
        return None
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return None

def test_existing_endpoints():
    """List existing endpoints to see what's available"""
    
    print("\n🔍 Checking existing endpoints...")
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        # Try to list existing endpoints
        response = requests.get(
            "https://api.runpod.io/v1/endpoints",
            headers=headers,
            timeout=15
        )
        
        print(f"List endpoints status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            endpoints = data.get('endpoints', []) if isinstance(data, dict) else data
            
            if endpoints:
                print(f"✅ Found {len(endpoints)} existing endpoints:")
                for ep in endpoints:
                    print(f"   ID: {ep.get('id', 'N/A')}")
                    print(f"   Name: {ep.get('name', 'N/A')}")
                    print(f"   Status: {ep.get('status', 'N/A')}")
                    print()
                    
                # Try to use the first available endpoint for testing
                if len(endpoints) > 0:
                    test_endpoint_id = endpoints[0].get('id')
                    if test_endpoint_id:
                        print(f"🧪 Testing existing endpoint: {test_endpoint_id}")
                        return test_existing_endpoint(test_endpoint_id)
            else:
                print(f"📝 No existing endpoints found")
        else:
            print(f"Response: {response.text}")
        
        return False
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False

def test_existing_endpoint(endpoint_id):
    """Test an existing endpoint with a simple payload"""
    
    print(f"🚀 Testing endpoint: {endpoint_id}")
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Simple test payload
    test_payload = {
        "input": {
            "test": "hello",
            "iterations": 5
        }
    }
    
    try:
        response = requests.post(
            f"https://api.runpod.ai/v2/{endpoint_id}/runsync",
            headers=headers,
            json=test_payload,
            timeout=60
        )
        
        print(f"Test response status: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Endpoint test successful!")
            return True
        else:
            print(f"❌ Endpoint test failed")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main testing function"""
    
    print("🎯 RUNPOD ENDPOINT API TESTING")
    print("="*35)
    
    # Test 1: Check existing endpoints
    existing_worked = test_existing_endpoints()
    
    if existing_worked:
        print(f"\n✅ SUCCESS: Found working endpoint!")
        return True
    
    # Test 2: Try to create new endpoint
    print(f"\n🏗️  Trying to create new endpoint...")
    endpoint_id = test_simple_endpoint_creation()
    
    if endpoint_id:
        print(f"\n✅ SUCCESS: Created new endpoint!")
        print(f"🎯 Endpoint ID: {endpoint_id}")
        return True
    
    print(f"\n❌ FAILED: Could not create or test endpoints")
    print(f"🎯 RECOMMENDATION: Use manual UI approach or Jupyter interface")
    
    return False

if __name__ == "__main__":
    main()