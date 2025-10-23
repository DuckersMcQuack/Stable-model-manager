#!/usr/bin/env python3
"""
Debug metadata.json parsing for base model extraction
"""

import os
import sys
import json

current_dir = os.getcwd()
sys.path.append(current_dir)

from metadata_extractor import MetadataExtractor

def test_metadata_json_parsing():
    """Test parsing of metadata.json for base model extraction"""
    
    # Create a sample metadata.json structure to test
    sample_metadata_json = {
        "base_model": "SD 1.5",
        "civitai": {
            "baseModel": "SDXL 1.0"
        },
        "other_field": "value"
    }
    
    # Test the extraction logic
    extractor = MetadataExtractor()
    
    # Simulate associated metadata structure
    associated_metadata = {
        'civitai_info': None,
        'metadata_json': {
            'path': 'sample_metadata/example_model.metadata.json',
            'data': sample_metadata_json
        },
        'readme_info': None,
        'other_text_files': []
    }
    
    model_metadata = {
        'file_path': 'sample_models/example_model.safetensors'
    }
    
    print("Testing metadata.json base model extraction:")
    print("=" * 50)
    print(f"Sample metadata.json: {json.dumps(sample_metadata_json, indent=2)}")
    
    result = extractor.extract_base_model_from_metadata(model_metadata, associated_metadata, skip_civitai_lookup=True)
    print(f"Extracted base model: {result}")
    
    # Test with just base_model field
    test_cases = [
        {"base_model": "SD 1.5"},
        {"civitai": {"baseModel": "SDXL 1.0"}},
        {"some_other_field": "value"},  # Should return None/Unknown
        None  # Empty metadata
    ]
    
    print("\nTesting various metadata.json structures:")
    print("=" * 50)
    
    for i, test_data in enumerate(test_cases, 1):
        test_associated = {
            'civitai_info': None,
            'metadata_json': {
                'path': f'sample_metadata/case{i}.metadata.json',
                'data': test_data
            },
            'readme_info': None,
            'other_text_files': []
        }
        
        result = extractor.extract_base_model_from_metadata(model_metadata, test_associated, skip_civitai_lookup=True)
        print(f"Case {i}: {test_data} → {result}")

if __name__ == "__main__":
    test_metadata_json_parsing()