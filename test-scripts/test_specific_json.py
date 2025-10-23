#!/usr/bin/env python3
"""
Test specific malformed JSON case
"""

import tempfile
import os
import sys

# Add parent directory to path for importing main modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from metadata_extractor import MetadataExtractor

def test_specific_case():
    """Test the specific 'Extra data: line 22 column 2' case"""
    
    # Simulate a JSON file with valid JSON followed by extra content
    content = '''{"model_name": "test-model", "base_model": "SD 1.5", "trained_words": ["word1", "word2"]}
Extra content on line 2
More extra content
And another line'''
    
    extractor = MetadataExtractor()
    
    print("Testing specific malformed JSON case...")
    print("=" * 50)
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.metadata.json', delete=False) as f:
        f.write(content)
        temp_path = f.name
    
    try:
        print(f"File content:")
        with open(temp_path, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines, 1):
                print(f"Line {i}: {line.rstrip()}")
        
        print(f"\nParsing result:")
        result = extractor.parse_metadata_json(temp_path)
        print(f"Parsed data: {result}")
        
        if result:
            print("✅ Successfully parsed despite extra content")
        else:
            print("❌ Failed to parse")
            
    finally:
        os.unlink(temp_path)

if __name__ == "__main__":
    test_specific_case()