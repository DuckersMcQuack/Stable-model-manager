#!/usr/bin/env python3
"""
Test malformed JSON parsing
"""

import json
import tempfile
import os
import sys

# Add parent directory to path for importing main modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from metadata_extractor import MetadataExtractor

def test_malformed_json():
    """Test parsing of malformed JSON files"""
    
    # Create test files with various JSON issues
    test_cases = [
        {
            'name': 'Valid JSON',
            'content': '{"model_name": "test", "base_model": "SD 1.5"}'
        },
        {
            'name': 'JSON with extra data',
            'content': '{"model_name": "test", "base_model": "SD 1.5"}\nExtra line\nAnother line'
        },
        {
            'name': 'JSON with trailing comma and extra data', 
            'content': '{"model_name": "test", "base_model": "SD 1.5",}\nExtra content here'
        }
    ]
    
    extractor = MetadataExtractor()
    
    print("Testing malformed JSON parsing...")
    print("=" * 50)
    
    for test_case in test_cases:
        print(f"\nTesting: {test_case['name']}")
        print(f"Content: {repr(test_case['content'])}")
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.metadata.json', delete=False) as f:
            f.write(test_case['content'])
            temp_path = f.name
        
        try:
            result = extractor.parse_metadata_json(temp_path)
            print(f"Result: {result}")
        finally:
            os.unlink(temp_path)

if __name__ == "__main__":
    test_malformed_json()