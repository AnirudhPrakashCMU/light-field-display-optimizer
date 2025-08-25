#!/usr/bin/env python3
"""
Simple GitHub Upload Test
Test GitHub upload locally before running on RunPod
"""

import requests
import json
import base64
from datetime import datetime

GITHUB_TOKEN = "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"
GITHUB_REPO = "AnirudhPrakashCMU/light-field-display-optimizer"

def test_github_upload():
    """Test GitHub upload with a simple file"""
    
    print("🧪 Testing GitHub upload...")
    
    # Create test content
    test_content = {
        'test': 'GitHub upload test from local machine',
        'timestamp': datetime.now().isoformat(),
        'message': 'Testing before RunPod optimization'
    }
    
    # Convert to base64
    content_str = json.dumps(test_content, indent=2)
    content_b64 = base64.b64encode(content_str.encode()).decode()
    
    # GitHub API request
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/test_results/upload_test_{datetime.now().strftime('%H%M%S')}.json"
    
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Content-Type": "application/json"
    }
    
    data = {
        "message": "Test GitHub upload from local machine",
        "content": content_b64
    }
    
    try:
        response = requests.put(url, headers=headers, json=data, timeout=30)
        
        print(f"Response status: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code in [200, 201]:
            print("✅ GitHub upload test SUCCESSFUL!")
            print(f"📁 File uploaded to: {url}")
            return True
        else:
            print(f"❌ GitHub upload FAILED: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Upload error: {e}")
        return False

if __name__ == "__main__":
    success = test_github_upload()
    
    if success:
        print("\n✅ GitHub upload is working!")
        print("🚀 Ready to run RunPod optimization with uploads")
    else:
        print("\n❌ GitHub upload is NOT working!")
        print("🔧 Fix the upload before running optimization")
        print("\nPossible issues:")
        print("- Token expired or invalid")
        print("- Token missing 'repo' permissions") 
        print("- Repository access denied")
        print("- Network connectivity issues")