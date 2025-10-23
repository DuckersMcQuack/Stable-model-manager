#!/usr/bin/env python3
"""
Test script for README parsing functionality
"""

import sys
import os

# Add parent directory to path for importing main modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from metadata_extractor import MetadataExtractor

def test_readme_parsing():
    """Test README.md parsing functionality"""
    extractor = MetadataExtractor()
    
    # Test with the 2091879 directory
    test_dir = "2091879"
    model_file = os.path.join(test_dir, "sample_video_model_v1.0.safetensors")
    
    if not os.path.exists(model_file):
        print(f"Test model file not found: {model_file}")
        return
    
    print("Testing README.md parsing...")
    print(f"Model file: {model_file}")
    
    # Test README parsing directly
    readme_path = os.path.join(test_dir, "README.md")
    if os.path.exists(readme_path):
        print(f"\nTesting direct README parsing: {readme_path}")
        readme_data = extractor.parse_readme_file(readme_path)
        print("README data:", readme_data)
    else:
        print("README.md not found")
        return
    
    # Test associated metadata finding
    print(f"\nTesting associated metadata finding...")
    associated_metadata = extractor.find_associated_metadata(model_file)
    print("Associated metadata keys:", list(associated_metadata.keys()))
    if associated_metadata['readme_info']:
        print("README info found:", associated_metadata['readme_info']['data'])
    else:
        print("No README info found")
    
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
        print("README data in comprehensive metadata:", comprehensive_metadata['readme_data'])

if __name__ == "__main__":
    test_readme_parsing()