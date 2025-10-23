#!/usr/bin/env python3
"""
Test script to verify README parsing works with filtered files
"""

import sys
import os

# Add parent directory to path for importing main modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from metadata_extractor import MetadataExtractor

def test_readme_filtering():
    """Test that README parsing still works after filtering"""
    extractor = MetadataExtractor()
    
    # Test with the new test directory
    test_dir = "test_readme_filter"
    model_file = os.path.join(test_dir, "example_model.safetensors")
    
    if not os.path.exists(model_file):
        print(f"Test model file not found: {model_file}")
        return
    
    print("Testing README parsing after filtering...")
    print(f"Model file: {model_file}")
    
    # Test comprehensive metadata extraction
    print(f"\nTesting comprehensive metadata extraction...")
    comprehensive_metadata = extractor.create_comprehensive_metadata(model_file)
    
    print(f"Model name: {comprehensive_metadata.get('model_name')}")
    print(f"Base model: {comprehensive_metadata.get('base_model')}")
    print(f"Model type: {comprehensive_metadata.get('model_type')}")
    print(f"Civitai ID: {comprehensive_metadata.get('civitai_id')}")
    print(f"Version ID: {comprehensive_metadata.get('version_id')}")
    print(f"Has README info: {comprehensive_metadata.get('has_readme_info')}")
    
    if comprehensive_metadata.get('readme_data'):
        print("README data found:", comprehensive_metadata['readme_data'])
    else:
        print("❌ No README data found - filtering may have broken README parsing")

if __name__ == "__main__":
    test_readme_filtering()