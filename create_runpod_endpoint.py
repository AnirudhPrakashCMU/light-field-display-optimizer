#!/usr/bin/env python3
"""
Create RunPod Serverless Endpoint via REST API
Programmatically create and test the light field optimizer endpoint
"""

import requests
import json
import time
from datetime import datetime

# Your RunPod API credentials
API_KEY = "rpa_LDVD1JMIAKGTVVO4AAV82Y5PN2FJTC2ROG5N6K6N1a1hn7"
GITHUB_REPO = "https://github.com/AnirudhPrakashCMU/light-field-display-optimizer"

def find_existing_template():
    """Find existing light field optimizer template"""
    
    print("🔍 Looking for existing light field optimizer template...")
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(
            "https://api.runpod.io/graphql",
            headers=headers,
            json={
                "query": """
                query getTemplates {
                    templates {
                        id
                        name
                        isServerless
                    }
                }
                """
            },
            timeout=15
        )
        
        if response.status_code == 200:
            data = response.json()
            templates = data.get('data', {}).get('templates', [])
            
            # Look for light field optimizer template
            for template in templates:
                if 'light-field-optimizer' in template.get('name', '').lower():
                    template_id = template.get('id')
                    print(f"✅ Found existing template: {template_id}")
                    print(f"   Name: {template.get('name')}")
                    return template_id
            
            print(f"❌ No existing light field template found")
            return None
            
        else:
            print(f"❌ Template search failed: {response.status_code}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Template search error: {e}")
        return None

def create_template():
    """Create a template for the light field optimizer"""
    
    print("🏗️  Creating RunPod template...")
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    template_data = {
        "name": f"light-field-optimizer-{datetime.now().strftime('%m%d-%H%M')}",
        "readme": "Enhanced Light Field Display Optimizer with Multi-Ray Sampling",
        "dockerArgs": "",
        "containerDiskInGb": 10,
        "volumeInGb": 0,
        "volumeMountPath": "",
        "ports": "",
        "startJupyter": False,
        "startSsh": False,
        "env": [
            {
                "key": "PYTORCH_CUDA_ALLOC_CONF",
                "value": "expandable_segments:True"
            },
            {
                "key": "CUDA_VISIBLE_DEVICES", 
                "value": "0"
            },
            {
                "key": "PYTHONUNBUFFERED",
                "value": "1"
            }
        ],
        "isPublic": True,
        "isServerless": True,
        "imageName": f"{GITHUB_REPO}#master"
    }
    
    try:
        response = requests.post(
            "https://api.runpod.io/graphql",
            headers=headers,
            json={
                "query": """
                mutation createTemplate($input: SaveTemplateInput!) {
                    saveTemplate(input: $input) {
                        id
                        name
                        isPublic
                        isServerless
                    }
                }
                """,
                "variables": {"input": template_data}
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            
            if 'errors' in data:
                error_msg = data['errors'][0].get('message', '')
                if 'must be unique' in error_msg:
                    print(f"⚠️  Template name conflict, trying to find existing template...")
                    return find_existing_template()
                else:
                    print(f"❌ Template creation failed: {data['errors']}")
                    return None
            
            template = data.get('data', {}).get('saveTemplate', {})
            if template:
                template_id = template.get('id')
                print(f"✅ Template created: {template_id}")
                print(f"   Name: {template.get('name')}")
                return template_id
            else:
                print(f"❌ No template data returned")
                return None
                
        else:
            print(f"❌ Template creation failed: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Template creation error: {e}")
        return None

def create_serverless_endpoint(template_id):
    """Create serverless endpoint using the template"""
    
    print(f"\n🚀 Creating serverless endpoint...")
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    endpoint_data = {
        "name": f"light-field-optimizer-{datetime.now().strftime('%m%d-%H%M')}",
        "templateId": template_id,
        "gpuIds": "NVIDIA A100 80GB PCIe",
        "scalerType": "QUEUE_DELAY",
        "scalerValue": 4,
        "workersMin": 0,
        "workersMax": 3,
        "idleTimeout": 5,
        "executionTimeoutMs": 3600000
    }
    
    try:
        response = requests.post(
            "https://api.runpod.io/graphql", 
            headers=headers,
            json={
                "query": """
                mutation saveEndpoint($input: EndpointInput!) {
                    saveEndpoint(input: $input) {
                        id
                        name
                        gpuIds
                    }
                }
                """,
                "variables": {"input": endpoint_data}
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            
            if 'errors' in data:
                print(f"❌ Endpoint creation failed: {data['errors']}")
                return None
            
            endpoint = data.get('data', {}).get('saveEndpoint', {})
            if endpoint:
                endpoint_id = endpoint.get('id')
                print(f"✅ Endpoint created: {endpoint_id}")
                print(f"   Name: {endpoint.get('name')}")
                print(f"   GPU IDs: {endpoint.get('gpuIds')}")
                return endpoint_id
            else:
                print(f"❌ No endpoint data returned")
                return None
                
        else:
            print(f"❌ Endpoint creation failed: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Endpoint creation error: {e}")
        return None

def wait_for_endpoint_ready(endpoint_id, max_wait_minutes=10):
    """Wait for endpoint to become ready"""
    
    print(f"\n⏳ Waiting for endpoint to become ready...")
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    start_time = time.time()
    
    while True:
        try:
            response = requests.post(
                "https://api.runpod.io/graphql",
                headers=headers,
                json={
                    "query": """
                    query getEndpoint($endpointId: String!) {
                        endpoint(id: $endpointId) {
                            id
                            name
                            status
                            locations {
                                id
                                status
                            }
                        }
                    }
                    """,
                    "variables": {"endpointId": endpoint_id}
                },
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                endpoint = data.get('data', {}).get('endpoint', {})
                
                if endpoint:
                    status = endpoint.get('status', 'UNKNOWN')
                    elapsed = (time.time() - start_time) / 60
                    
                    print(f"   [{elapsed:.1f}m] Status: {status}")
                    
                    if status == 'READY':
                        print(f"✅ Endpoint is ready!")
                        return True
                    elif status in ['FAILED', 'TERMINATED']:
                        print(f"❌ Endpoint failed: {status}")
                        return False
                        
            else:
                print(f"❌ Status check failed: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Status check error: {e}")
        
        # Check timeout
        elapsed = (time.time() - start_time) / 60
        if elapsed >= max_wait_minutes:
            print(f"⏰ Timeout after {max_wait_minutes} minutes")
            return False
        
        time.sleep(30)  # Wait 30 seconds between checks

def test_endpoint(endpoint_id):
    """Test the endpoint with light field optimization"""
    
    print(f"\n🧪 Testing endpoint with light field optimization...")
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Test 1: Quick test
    print("   Running quick checkerboard test...")
    test_payload = {
        "input": {
            "task_type": "quick_test",
            "iterations": 20,
            "resolution": 256,
            "rays_per_pixel": 8,
            "job_id": f"test_{datetime.now().strftime('%H%M%S')}"
        }
    }
    
    try:
        response = requests.post(
            f"https://api.runpod.ai/v2/{endpoint_id}/runsync",
            headers=headers,
            json=test_payload,
            timeout=300  # 5 minute timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            
            if 'error' in result:
                print(f"❌ Endpoint execution error: {result['error']}")
                return False
            
            output = result.get('output', {})
            if output.get('status') == 'success':
                results = output.get('results', {})
                print(f"✅ Quick test successful!")
                print(f"   Final loss: {results.get('final_loss', 'N/A')}")
                print(f"   Iterations: {results.get('iterations', 'N/A')}")
                print(f"   GPU memory used: {results.get('gpu_memory_used', 'N/A'):.2f} GB")
                
                # Test 2: Full optimization (shorter)
                print(f"\n   Running full optimization test...")
                full_payload = {
                    "input": {
                        "task_type": "full_optimization",
                        "iterations": 50,
                        "resolution": 384,
                        "scenes": ["spherical_checkerboard"],
                        "save_large_outputs": False
                    }
                }
                
                full_response = requests.post(
                    f"https://api.runpod.ai/v2/{endpoint_id}/runsync",
                    headers=headers,
                    json=full_payload,
                    timeout=600  # 10 minute timeout
                )
                
                if full_response.status_code == 200:
                    full_result = full_response.json()
                    full_output = full_result.get('output', {})
                    
                    if full_output.get('status') == 'success':
                        print(f"✅ Full optimization successful!")
                        full_results = full_output.get('results', {})
                        print(f"   Scenes completed: {full_results.get('scenes_completed', [])}")
                        print(f"   Peak GPU memory: {full_results.get('gpu_memory_peak', 'N/A'):.2f} GB")
                        return True
                    else:
                        print(f"❌ Full optimization failed: {full_output.get('message', 'Unknown error')}")
                        return False
                else:
                    print(f"❌ Full optimization request failed: {full_response.status_code}")
                    return False
                
            else:
                print(f"❌ Quick test failed: {output.get('message', 'Unknown error')}")
                return False
                
        else:
            print(f"❌ Test request failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Test request error: {e}")
        return False

def main():
    """Main endpoint creation and testing function"""
    
    print("🎯 RUNPOD SERVERLESS ENDPOINT CREATION VIA API")
    print("="*55)
    
    # Step 1: Create template
    template_id = create_template()
    if not template_id:
        print(f"\n❌ FAILED: Could not create template")
        return False
    
    # Step 2: Create endpoint
    endpoint_id = create_serverless_endpoint(template_id)
    if not endpoint_id:
        print(f"\n❌ FAILED: Could not create endpoint")
        return False
    
    # Step 3: Wait for endpoint to be ready
    ready = wait_for_endpoint_ready(endpoint_id, max_wait_minutes=15)
    if not ready:
        print(f"\n❌ FAILED: Endpoint not ready within timeout")
        return False
    
    # Step 4: Test the endpoint
    test_success = test_endpoint(endpoint_id)
    
    if test_success:
        print(f"\n🎉 SUCCESS: Endpoint created and tested successfully!")
        print(f"🚀 Endpoint ID: {endpoint_id}")
        print(f"📡 API URL: https://api.runpod.ai/v2/{endpoint_id}")
        print(f"🔗 GitHub Repo: {GITHUB_REPO}")
        
        print(f"\n📝 Usage Example:")
        print(f'curl -X POST "https://api.runpod.ai/v2/{endpoint_id}/runsync" \\')
        print(f'  -H "Authorization: Bearer {API_KEY}" \\')
        print(f'  -H "Content-Type: application/json" \\')
        print(f'  -d \'{{"input": {{"task_type": "quick_test", "iterations": 50}}}}\'')
        
        return True
    else:
        print(f"\n❌ FAILED: Endpoint created but testing failed")
        print(f"🔧 Endpoint ID: {endpoint_id} (for debugging)")
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print(f"\n✅ Light Field Optimizer endpoint is live and working!")
    else:
        print(f"\n❌ Endpoint creation or testing failed. Check logs above.")