#!/usr/bin/env python3
"""
Test script to verify AutoV3 hash computation
"""

import sys
import os
from pathlib import Path
from test_utils import setup_parent_import, get_safetensors_files, print_available_files, get_first_safetensors

# Setup import path for parent directory
setup_parent_import()
from file_scanner import FileScanner

def test_autov3_hashes():
    """Test AutoV3 hash computation on SafeTensors files"""
    scanner = FileScanner()
    
    print("🔍 Testing AutoV3 hash computation...")
    print("=" * 60)
    
    # Show available files
    print_available_files()
    
    # Get SafeTensors files from current test directory
    safetensors_files = get_safetensors_files()
    
    if not safetensors_files:
        return
    
    print()
    
    for filename in safetensors_files[:5]:  # Test first 5 files
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
        print(f"📁 File: {filename}")
        
        try:
            sha256, autov3, blake3 = scanner.calculate_hashes(file_path)
            print(f"  📋 SHA256:  {sha256}")
            print(f"  🔥 AutoV3:  {autov3 if autov3 else 'N/A (not SafeTensors or error)'}")
            
            if autov3:
                print(f"  ✅ AutoV3 (first 12 chars): {autov3[:12]}")
        except Exception as e:
            print(f"  ❌ Error: {e}")
        
        print()

if __name__ == "__main__":
    test_autov3_hashes()