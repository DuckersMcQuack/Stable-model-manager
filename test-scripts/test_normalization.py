#!/usr/bin/env python3
"""
Test the normalization function with various inputs to validate database integration.
"""

import os
import sys

# Add current directory to path
current_dir = os.getcwd()
sys.path.append(current_dir)

from metadata_extractor import MetadataExtractor

def test_normalization():
    """Test the dynamic base model normalization"""
    
    extractor = MetadataExtractor()
    
    test_cases = [
        # Direct database matches
        ("SD 1.5", "SD 1.5"),
        ("SDXL 1.0", "SDXL 1.0"),
        ("Flux.1 D", "Flux.1 D"),
        ("Pony", "Pony"),
        ("Illustrious", "Illustrious"),
        ("Other", "Other"),
        
        # Common variations that should be normalized
        ("sdxl", "SDXL 1.0"),
        ("SD XL", "SDXL 1.0"),
        ("stable diffusion xl", "SDXL 1.0"),
        ("flux", "Flux.1 D"),
        ("FLUX.1", "Flux.1 D"),
        ("sd1.5", "SD 1.5"),
        ("stable diffusion 1.5", "SD 1.5"),
        ("sd 2.1", "SD 2.1 768"),  # Should match closest
        ("ponyxl", "Pony"),
        
        # Edge cases
        ("unknown_model", "Unknown_Model"),  # Should title case unknown
        ("", "Other"),  # Empty should be Other
        (None, "Other"),  # None should be Other
    ]
    
    print("Testing dynamic base model normalization:")
    print("=" * 60)
    
    for input_val, expected in test_cases:
        result = extractor.normalize_base_model_name(input_val)
        status = "✓" if result == expected else "?"
        print(f"{status} Input: '{input_val}' → Output: '{result}'")
        if result != expected:
            print(f"   Expected: '{expected}'")
        print()
    
    # Show what base models are available in the database
    print("\nValid base models from civitai database:")
    print("=" * 60)
    valid_models = extractor._get_valid_base_models()
    for i, model in enumerate(valid_models[:20], 1):  # Show first 20
        print(f"{i:2d}. {model}")
    if len(valid_models) > 20:
        print(f"... and {len(valid_models) - 20} more")

if __name__ == "__main__":
    test_normalization()