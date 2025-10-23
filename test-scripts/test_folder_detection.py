#!/usr/bin/env python3
"""
Test script for folder-based base model detection functionality.
"""

import os
import sys

# Add current directory to path
current_dir = os.getcwd()
sys.path.append(current_dir)

from metadata_extractor import MetadataExtractor

def test_folder_detection():
    """Test the folder-based base model detection"""
    
    extractor = MetadataExtractor()
    
    # First test with predefined patterns
    test_cases = [
        # Test various folder patterns
        ("/path/to/SD 1.5/model.safetensors", "SD 1.5"),
        ("/path/to/SDXL/sample_model.safetensors", "SDXL 1.0"),
        ("/path/to/FLUX.1/sample_model.safetensors", "Flux.1 D"),
        ("/path/to/Pony/sample_model.safetensors", "Pony"),
        ("/path/to/SD3/sample_model.safetensors", "SD 3"),
        ("/path/to/illustrious/sample_model.safetensors", "Illustrious"),
        ("/path/to/SD2.1/sample_model.safetensors", "SD 2.1"),
        ("/path/to/random_folder/model.safetensors", None),
        # Test nested patterns
        ("/models/Stable Diffusion/SD 1.5/Anime/model.safetensors", "SD 1.5"),
        ("/loras/FLUX.1-dev/characters/sample_model.safetensors", "Flux.1 D"),
    ]
    
    # Add actual files from sample_models directory
    real_files = []
    if os.path.exists('sample_models'):
        for f in os.listdir('sample_models'):
            if f.lower().endswith(('.safetensors', '.ckpt', '.pt', '.pth')) and os.path.isfile(os.path.join('sample_models', f)):
                full_path = os.path.abspath(os.path.join('sample_models', f))
                real_files.append((full_path, "Test with real file"))
    
    all_test_cases = test_cases + real_files
    
    print("Testing folder-based base model detection:")
    print("=" * 60)
    
    if real_files:
        print(f"Found {len(real_files)} actual model files in sample_models/ for testing")
    else:
        print("No model files found in sample_models/ - testing with synthetic paths only")
        print("💡 Add .safetensors/.ckpt files to sample_models/ for real file testing")
    print()
    
    for file_path, expected in all_test_cases:
        result = extractor.extract_base_model_from_path(file_path)
        status = "✓" if result == expected else "✗"
        print(f"{status} Path: {file_path}")
        print(f"   Expected: {expected}")
        print(f"   Got: {result}")
        print()
    
    # Test the full metadata extraction with folder fallback
    print("\nTesting full metadata extraction with folder fallback:")
    print("=" * 60)
    
    # Simulate a model with "Other" base model that should use folder detection
    test_model = {
        'file_path': '/models/SDXL/sample_lora.safetensors',
        'filename': 'sample_lora.safetensors'
    }
    
    test_associated = {
        'civitai_info': None,
        'metadata_json': None,
        'readme_info': None,
        'other_text_files': [],
        'ss_base_model_version': 'Other'  # This should trigger folder detection
    }
    
    # Test with skip_civitai_lookup = True (model not in civitai database)
    result = extractor.extract_base_model_from_metadata(
        test_model, test_associated, skip_civitai_lookup=True
    )
    
    print(f"Model path: {test_model['file_path']}")
    print(f"Metadata base model: {test_associated['ss_base_model_version']}")
    print(f"Skip civitai lookup: True")
    print(f"Final base model: {result}")
    print(f"Expected: SDXL 1.0")
    print(f"Status: {'✓' if result == 'SDXL 1.0' else '✗'}")

if __name__ == "__main__":
    test_folder_detection()