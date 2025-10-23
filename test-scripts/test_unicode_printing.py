#!/usr/bin/env python3
"""
Test Unicode filename handling
"""

import sys
import os

# Add parent directory to path for importing main modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from metadata_extractor import MetadataExtractor

def test_unicode_printing():
    """Test Unicode filename printing"""
    extractor = MetadataExtractor()
    
    # Test various Unicode filenames
    test_paths = [
        "/path/to/models/SD 1.5/unicode_test_女孩/model.safetensors",
        "/path/to/models/SD 1.5/unicode_test_时之界/model.safetensors", 
        "/path/to/models/SD 1.5/unicode_test_日本語/model.safetensors",
        "/path/to/models/regular_ascii_path/model.safetensors"
    ]
    
    print("Testing Unicode filename printing...")
    print("=" * 50)
    
    for path in test_paths:
        print(f"\nTesting path: {path}")
        extractor.safe_print_path("Processing", path)
        extractor.safe_print_path("Skipping", path)

if __name__ == "__main__":
    test_unicode_printing()