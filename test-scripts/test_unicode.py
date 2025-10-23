#!/usr/bin/env python3
"""
Test Unicode path handling
"""

import os
import sys

# Add current directory to path
current_dir = os.getcwd()
sys.path.append(current_dir)

from metadata_extractor import MetadataExtractor

def test_unicode_paths():
    """Test Unicode path printing"""
    
    extractor = MetadataExtractor()
    
    # Test paths with various Unicode characters
    test_paths = [
        "sample_models/example_model.safetensors",
        "sample_models/unicode_test_model.safetensors", 
        "sample_models/japanese_文字.safetensors", 
        "sample_models/chinese_中文.safetensors",
        "sample_models/korean_한국어.safetensors", 
        "sample_models/emoji_😀😃😄.safetensors",
        "sample_models/special_çñüñ.safetensors",
    ]
    
    print("Testing Unicode path handling:")
    print("=" * 60)
    
    for path in test_paths:
        print(f"\nTesting path: {repr(path)}")
        extractor.safe_print_path("Test message", path)
    
    print("\n" + "=" * 60)
    print("Terminal encoding info:")
    print(f"sys.stdout.encoding: {sys.stdout.encoding}")
    print(f"sys.getdefaultencoding(): {sys.getdefaultencoding()}")
    
    # Also test the specific problematic path
    problematic_path = "sample_models/unicode_test_model.safetensors"
    print(f"\nSpecific problematic path test:")
    extractor.safe_print_path("Skipping (found in civitai database)", problematic_path)

if __name__ == "__main__":
    test_unicode_paths()