#!/usr/bin/env python3
"""
Local Upload Test Script
Test file upload services locally before deploying to RunPod
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import requests
import json
from datetime import datetime

print("🧪 LOCAL UPLOAD TEST")

def create_test_checkerboard():
    """Create a test checkerboard image"""
    
    print("🎨 Creating test checkerboard...")
    
    resolution = 256
    checkerboard = np.zeros((resolution, resolution, 3))
    
    # Create checkerboard pattern
    square_size = 32
    for i in range(0, resolution, square_size):
        for j in range(0, resolution, square_size):
            if (i // square_size + j // square_size) % 2 == 0:
                checkerboard[i:i+square_size, j:j+square_size] = 1.0
    
    # Save image
    plt.figure(figsize=(6, 6))
    plt.imshow(checkerboard)
    plt.title('Local Upload Test - Checkerboard Pattern')
    plt.axis('off')
    
    test_image_path = 'test_checkerboard.png'
    plt.savefig(test_image_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Test image created: {test_image_path}")
    return test_image_path

def test_0x0_upload(file_path):
    """Test upload to 0x0.st"""
    
    print(f"\n📤 Testing 0x0.st upload...")
    
    try:
        filename = os.path.basename(file_path)
        file_size = os.path.getsize(file_path) / 1024**2
        
        print(f"   File: {filename} ({file_size:.1f} MB)")
        
        with open(file_path, 'rb') as f:
            files = {'file': (filename, f)}
            response = requests.post('https://0x0.st', files=files, timeout=60)
        
        print(f"   Response status: {response.status_code}")
        print(f"   Response content: {response.text[:100]}")
        
        if response.status_code == 200:
            url = response.text.strip()
            if url.startswith('https://'):
                print(f"✅ 0x0.st upload SUCCESSFUL!")
                print(f"🔗 Download URL: {url}")
                return url
            else:
                print(f"❌ Invalid URL response: {response.text}")
        else:
            print(f"❌ Upload failed with status {response.status_code}")
    
    except Exception as e:
        print(f"❌ 0x0.st upload error: {e}")
    
    return None

def test_fileio_upload(file_path):
    """Test upload to file.io"""
    
    print(f"\n📤 Testing file.io upload...")
    
    try:
        filename = os.path.basename(file_path)
        file_size = os.path.getsize(file_path) / 1024**2
        
        print(f"   File: {filename} ({file_size:.1f} MB)")
        
        with open(file_path, 'rb') as f:
            files = {'file': f}
            response = requests.post('https://file.io', files=files, timeout=60)
        
        print(f"   Response status: {response.status_code}")
        print(f"   Response content: {response.text[:200]}")
        
        if response.status_code == 200:
            try:
                result = response.json()
                print(f"   JSON response: {result}")
                
                if result.get('success'):
                    url = result.get('link')
                    print(f"✅ file.io upload SUCCESSFUL!")
                    print(f"🔗 Download URL: {url}")
                    return url
                else:
                    print(f"❌ file.io reported failure: {result}")
            except json.JSONDecodeError:
                print(f"❌ file.io returned HTML instead of JSON (likely blocked/rate limited)")
                print(f"   HTML content: {response.text[:300]}")
        else:
            print(f"❌ Upload failed with status {response.status_code}")
    
    except Exception as e:
        print(f"❌ file.io upload error: {e}")
    
    return None

def test_imgur_upload(file_path):
    """Test upload to Imgur (requires no API key for anonymous uploads)"""
    
    print(f"\n📤 Testing Imgur upload...")
    
    try:
        filename = os.path.basename(file_path)
        file_size = os.path.getsize(file_path) / 1024**2
        
        print(f"   File: {filename} ({file_size:.1f} MB)")
        
        with open(file_path, 'rb') as f:
            files = {'image': f}
            headers = {'Authorization': 'Client-ID 546c25a59c58ad7'}  # Anonymous client ID
            response = requests.post('https://api.imgur.com/3/image', files=files, headers=headers, timeout=60)
        
        print(f"   Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                url = result.get('data', {}).get('link')
                print(f"✅ Imgur upload SUCCESSFUL!")
                print(f"🔗 Download URL: {url}")
                return url
            else:
                print(f"❌ Imgur reported failure: {result}")
        else:
            print(f"❌ Imgur upload failed: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
    
    except Exception as e:
        print(f"❌ Imgur upload error: {e}")
    
    return None

def test_catbox_upload(file_path):
    """Test upload to catbox.moe"""
    
    print(f"\n📤 Testing catbox.moe upload...")
    
    try:
        filename = os.path.basename(file_path)
        file_size = os.path.getsize(file_path) / 1024**2
        
        print(f"   File: {filename} ({file_size:.1f} MB)")
        
        with open(file_path, 'rb') as f:
            files = {'fileToUpload': f}
            data = {'reqtype': 'fileupload'}
            response = requests.post('https://catbox.moe/user/api.php', files=files, data=data, timeout=60)
        
        print(f"   Response status: {response.status_code}")
        print(f"   Response content: {response.text}")
        
        if response.status_code == 200:
            url = response.text.strip()
            if url.startswith('https://'):
                print(f"✅ catbox.moe upload SUCCESSFUL!")
                print(f"🔗 Download URL: {url}")
                return url
            else:
                print(f"❌ Invalid catbox response: {response.text}")
        else:
            print(f"❌ catbox upload failed: {response.status_code}")
    
    except Exception as e:
        print(f"❌ catbox upload error: {e}")
    
    return None

def main():
    """Main test function"""
    
    print("🔬 TESTING FILE UPLOAD SERVICES LOCALLY")
    print("="*50)
    
    # Create test image
    test_image = create_test_checkerboard()
    
    # Test all upload services
    upload_results = {}
    
    # Test 1: 0x0.st
    url_0x0 = test_0x0_upload(test_image)
    upload_results['0x0.st'] = url_0x0
    
    # Test 2: file.io
    url_fileio = test_fileio_upload(test_image)
    upload_results['file.io'] = url_fileio
    
    # Test 3: Imgur
    url_imgur = test_imgur_upload(test_image)
    upload_results['imgur'] = url_imgur
    
    # Test 4: catbox.moe
    url_catbox = test_catbox_upload(test_image)
    upload_results['catbox.moe'] = url_catbox
    
    # Summary
    print(f"\n" + "="*60)
    print("📊 UPLOAD TEST SUMMARY:")
    print("="*60)
    
    working_services = []
    for service, url in upload_results.items():
        if url:
            print(f"✅ {service}: {url}")
            working_services.append(service)
        else:
            print(f"❌ {service}: FAILED")
    
    print("="*60)
    print(f"📊 Working services: {len(working_services)}/4")
    
    if working_services:
        print(f"🎯 RECOMMENDED: Use {working_services[0]} for RunPod uploads")
        
        # Update RunPod handler recommendation
        print(f"\n📝 UPDATE RUNPOD HANDLER:")
        if working_services[0] == '0x0.st':
            print("   Use upload_to_0x0() function")
        elif working_services[0] == 'file.io':
            print("   Use upload_to_fileio() function")
        elif working_services[0] == 'imgur':
            print("   Use upload_to_imgur() function")
        elif working_services[0] == 'catbox.moe':
            print("   Use upload_to_catbox() function")
    else:
        print("❌ ALL UPLOAD SERVICES FAILED!")
        print("🔧 Consider alternative storage or local file saving")
    
    # Clean up
    if os.path.exists(test_image):
        os.remove(test_image)
    
    return working_services

if __name__ == "__main__":
    working = main()
    
    if working:
        print(f"\n✅ Upload testing complete - {len(working)} services working")
        print(f"🚀 Ready to update RunPod handler with working upload service")
    else:
        print(f"\n❌ No working upload services found")
        print(f"🔧 Need to find alternative storage solution")