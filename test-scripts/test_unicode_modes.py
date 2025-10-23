#!/usr/bin/env python3
"""
Test Unicode handling with verbose/non-verbose modes
"""

import os
import sys

current_dir = os.getcwd()
sys.path.append(current_dir)

from metadata_extractor import MetadataExtractor

def test_unicode_modes():
    """Test Unicode handling with different verbosity levels"""
    
    print("Testing VERBOSE mode:")
    print("=" * 50)
    
    # Test verbose mode
    extractor_verbose = MetadataExtractor(verbose=True)
    
    problematic_paths = [
        "sample_models/unicode_test_达克尼斯_为美好的世界献上祝福.safetensors",
        "/path/with/chinese/中文模型.safetensors",
        "/path/with/japanese/日本語モデル.safetensors",
        "/very/long/path/that/might/cause/wrapping/issues/with/unicode/characters/中文/日本語/한국어/достаточно/длинный/путь.safetensors"
    ]
    
    for path in problematic_paths:
        extractor_verbose.safe_print_path("Testing verbose", path)
    
    print("\nTesting NON-VERBOSE mode:")
    print("=" * 50)
    
    # Test non-verbose mode  
    extractor_quiet = MetadataExtractor(verbose=False)
    
    for path in problematic_paths:
        extractor_quiet.safe_print_path("Testing quiet", path)
    
    print("Non-verbose mode should show no output above this line.")
    
    print("\nTesting safe_path_repr function:")
    print("=" * 50)
    
    for path in problematic_paths:
        safe_repr = extractor_verbose.safe_path_repr(path)
        print(f"Original length: {len(path)}, Safe length: {len(safe_repr)}")
        print(f"Safe repr: {safe_repr}")
        print()

if __name__ == "__main__":
    test_unicode_modes()