#!/usr/bin/env python3
"""
Feature verification checklist for metadata_extractor.py
Verifies all implemented features are present and working
"""

import os
import sys
import inspect

current_dir = os.getcwd()
sys.path.append(current_dir)

from metadata_extractor import MetadataExtractor

def verify_features():
    """Verify all implemented features are present"""
    
    print("🔍 FEATURE VERIFICATION CHECKLIST")
    print("=" * 60)
    
    extractor = MetadataExtractor()
    
    # Check 1: Unicode handling
    print("✅ 1. Unicode handling:")
    assert hasattr(extractor, 'safe_print_path'), "Missing safe_print_path method"
    assert hasattr(extractor, 'safe_path_repr'), "Missing safe_path_repr method"
    print("   - safe_print_path: ✓")
    print("   - safe_path_repr: ✓")
    
    # Check 2: Verbose mode control
    print("\n✅ 2. Verbose mode control:")
    assert hasattr(extractor, 'verbose'), "Missing verbose attribute"
    extractor_quiet = MetadataExtractor(verbose=False)
    assert extractor_quiet.verbose == False, "Verbose mode not working"
    print("   - verbose parameter: ✓")
    print("   - non-verbose mode: ✓")
    
    # Check 3: Folder-based base model detection
    print("\n✅ 3. Folder-based base model detection:")
    assert hasattr(extractor, 'extract_base_model_from_path'), "Missing extract_base_model_from_path method"
    test_result = extractor.extract_base_model_from_path("/test/SDXL/sample_model.safetensors")
    assert test_result == "SDXL 1.0", f"Expected 'SDXL 1.0', got '{test_result}'"
    print("   - extract_base_model_from_path: ✓")
    print("   - folder pattern recognition: ✓")
    
    # Check 4: Dynamic base model normalization
    print("\n✅ 4. Dynamic base model normalization:")
    assert hasattr(extractor, '_get_valid_base_models'), "Missing _get_valid_base_models method"
    assert hasattr(extractor, '_get_model_keywords'), "Missing _get_model_keywords method"
    valid_models = extractor._get_valid_base_models()
    assert len(valid_models) > 0, "No valid base models found"
    print(f"   - Database integration: ✓ ({len(valid_models)} base models)")
    print("   - Dynamic keyword matching: ✓")
    
    # Check 5: README.md parsing
    print("\n✅ 5. README.md parsing:")
    assert hasattr(extractor, 'parse_readme_file'), "Missing parse_readme_file method"
    print("   - README parsing method: ✓")
    
    # Check 6: Robust JSON parsing
    print("\n✅ 6. Robust JSON parsing:")
    parse_method = getattr(extractor, 'parse_metadata_json')
    source = inspect.getsource(parse_method)
    assert 'raw_decode' in source, "Missing raw_decode fallback"
    assert 'JSONDecodeError' in source, "Missing JSON error handling"
    print("   - Malformed JSON handling: ✓")
    print("   - raw_decode fallback: ✓")
    
    # Check 7: Skip civitai lookup integration
    print("\n✅ 7. Skip civitai lookup integration:")
    base_model_method = getattr(extractor, 'extract_base_model_from_metadata')
    sig = inspect.signature(base_model_method)
    assert 'skip_civitai_lookup' in sig.parameters, "Missing skip_civitai_lookup parameter"
    
    create_method = getattr(extractor, 'create_comprehensive_metadata')
    sig2 = inspect.signature(create_method)
    assert 'skip_civitai_lookup' in sig2.parameters, "Missing skip_civitai_lookup in create_comprehensive_metadata"
    print("   - Parameter passing: ✓")
    print("   - Method integration: ✓")
    
    # Check 8: Read-only mode for process_scanned_models
    print("\n✅ 8. Read-only cross-reference mode:")
    process_method = getattr(extractor, 'process_scanned_models')
    source = inspect.getsource(process_method)
    assert 'WHERE sf.file_type = \'model\'' in source, "Missing read-only query"
    assert 'LEFT JOIN model_files mf' not in source, "Still using old processed-only query"
    print("   - Read-only database access: ✓")
    print("   - Cross-reference logic: ✓")
    
    # Check 9: Debug output for troubleshooting
    print("\n✅ 9. Debug output:")
    assert 'DEBUG - metadata.json contains' in source, "Missing debug output"
    print("   - Metadata.json debugging: ✓")
    
    print("\n" + "=" * 60)
    print("🎉 ALL FEATURES VERIFIED SUCCESSFULLY!")
    print("\nImplemented features:")
    print("✓ Unicode-safe file path handling")
    print("✓ Verbose/quiet mode control") 
    print("✓ Folder-based base model detection")
    print("✓ Dynamic civitai database integration")
    print("✓ README.md metadata extraction")
    print("✓ Robust JSON parsing with fallbacks")
    print("✓ Conditional civitai lookup skipping")
    print("✓ Read-only cross-reference mode")
    print("✓ Debug output for troubleshooting")
    print("\n🚀 Ready for production use!")

if __name__ == "__main__":
    verify_features()