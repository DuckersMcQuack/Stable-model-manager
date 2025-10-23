#!/usr/bin/env python3
"""
Test script for advanced metadata extraction and cross-referencing functions
Tests all new functionality with sample directories (sample_models/, sample_media/, sample_metadata/)
"""

import os
import sys
import json
import sqlite3
from pathlib import Path

# Add parent directory to path for importing main modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import our enhanced file scanner
from file_scanner import (
    FileScanner, 
    extract_comprehensive_metadata,
    extract_image_metadata,
    parse_generation_parameters,
    read_metadata_text_file,
    find_metadata_text_file,
    cross_reference_with_civitai,
    get_default_config
)

def test_metadata_extraction():
    """Test comprehensive metadata extraction on sample files"""
    print("🔍 Testing Metadata Extraction Functions")
    print("=" * 50)
    
    # Dynamically find test files in sample directories
    sample_dirs = {
        'models': 'sample_models',
        'media': 'sample_media', 
        'metadata': 'sample_metadata'
    }
    
    test_files = []
    
    # Scan sample_media for image files to test metadata extraction
    media_dir = sample_dirs['media']
    if os.path.exists(media_dir):
        for ext in ['.png', '.jpg', '.jpeg', '.webp', '.gif']:
            test_files.extend([os.path.join(media_dir, f) for f in os.listdir(media_dir) 
                              if f.lower().endswith(ext) and os.path.isfile(os.path.join(media_dir, f))])
    
    # If no files found, create example message
    if not test_files:
        print("   ⚠️  No media files found in sample_media/ directory")
        print("   💡 Add .png, .jpg, .webp files to sample_media/ to test metadata extraction")
        return
    
    print(f"   📁 Found {len(test_files)} media files to test")
    
    for file_path in test_files:
        if os.path.exists(file_path):
            print(f"\n📁 Testing: {os.path.basename(file_path)}")
            try:
                # First extract basic metadata
                basic_metadata = extract_image_metadata(file_path)
                
                # Then extract comprehensive metadata
                metadata = extract_comprehensive_metadata(file_path, basic_metadata)
                
                print(f"   ✅ Metadata extracted successfully")
                print(f"   📐 Dimensions: {metadata.get('width', 'N/A')}x{metadata.get('height', 'N/A')}")
                print(f"   🎨 Generation Tool: {metadata.get('generation_tool', 'N/A')}")
                print(f"   🔢 Steps: {metadata.get('steps', 'N/A')}")
                print(f"   ⚙️ Sampler: {metadata.get('sampler', 'N/A')}")
                print(f"   🎯 CFG Scale: {metadata.get('cfg_scale', 'N/A')}")
                print(f"   🌱 Seed: {metadata.get('seed', 'N/A')}")
                print(f"   🔑 Model Hash: {metadata.get('model_hash', 'N/A')}")
                print(f"   🏷️ Base Model: {metadata.get('base_model', 'N/A')}")
                print(f"   📝 Raw Parameters: {len(metadata.get('raw_parameters', '') or '') > 0}")
                
                # Test component detection
                components = metadata.get('components', [])
                if components:
                    print(f"   🧩 Found {len(components)} components:")
                    for comp in components[:3]:  # Show first 3
                        print(f"      - {comp['type']}: {comp['name']} (weight: {comp.get('weight', 'N/A')})")
                else:
                    print(f"   🧩 No components detected")
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
        else:
            print(f"\n📁 File not found: {file_path}")

def test_generation_parameter_parsing():
    """Test generation parameter parsing with sample strings"""
    print("\n\n🎛️ Testing Generation Parameter Parsing")
    print("=" * 50)
    
    # Sample parameter strings to test
    test_strings = [
        "Steps: 30, Sampler: Euler a, CFG scale: 9.0, Seed: 123456789, Size: 512x768, Model hash: abc123def456, Model: sample_model_v1",
        "masterpiece, best quality, <lora:choco-pynoise-000012:1.0>, detailed face, <lora:add_detail:0.5>",
        "Steps: 25, Sampler: DPM++ 2M Karras, CFG scale: 7.5, Seed: 987654321, Model: deliberate_v2, VAE: vae-ft-mse-840000",
        "<lora:style_slider_v1:0.8>, anime style, <lyco:background_enhancer:0.6>, detailed background"
    ]
    
    for i, param_string in enumerate(test_strings, 1):
        print(f"\n📋 Test String {i}:")
        print(f"   Input: {param_string[:80]}{'...' if len(param_string) > 80 else ''}")
        
        try:
            result = parse_generation_parameters(param_string)
            
            print(f"   ✅ Parsed successfully")
            print(f"   🔢 Steps: {result.get('steps', 'Not found')}")
            print(f"   ⚙️ Sampler: {result.get('sampler', 'Not found')}")
            print(f"   🎯 CFG: {result.get('cfg_scale', 'Not found')}")
            print(f"   🌱 Seed: {result.get('seed', 'Not found')}")
            print(f"   🔑 Model Hash: {result.get('model_hash', 'Not found')}")
            
            components = result.get('components', [])
            if components:
                print(f"   🧩 Components found: {len(components)}")
                for comp in components:
                    print(f"      - {comp['type']}: {comp['name']} @ {comp.get('weight', 1.0)}")
            else:
                print(f"   🧩 No components found")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")

def test_metadata_text_files():
    """Test text metadata file detection and parsing"""
    print("\n\n📄 Testing Metadata Text File Functions")
    print("=" * 50)
    
    # Find actual files to test with
    test_images = []
    
    # Look in sample_media for image files
    if os.path.exists('sample_media'):
        for f in os.listdir('sample_media'):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')) and os.path.isfile(os.path.join('sample_media', f)):
                test_images.append(os.path.join('sample_media', f))
    
    # Look in sample_models for model preview images
    if os.path.exists('sample_models'):
        for f in os.listdir('sample_models'):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')) and os.path.isfile(os.path.join('sample_models', f)):
                test_images.append(os.path.join('sample_models', f))
    
    if not test_images:
        print("   ⚠️  No image files found in sample directories")
        print("   💡 Add image files to sample_media/ or sample_models/ to test text file detection")
        return
    
    print(f"   📁 Testing {len(test_images)} image files for associated text files")
    
    for image_path in test_images:
        if os.path.exists(image_path):
            print(f"\n📁 Testing: {os.path.basename(image_path)}")
            
            # Test finding metadata text file
            text_file = find_metadata_text_file(image_path)
            if text_file:
                print(f"   ✅ Found text file: {os.path.basename(text_file)}")
                
                # Test reading the text file
                try:
                    text_metadata = read_metadata_text_file(text_file)
                    print(f"   📝 Text metadata extracted:")
                    print(f"      Model references: {len(text_metadata.get('model_references', []))}")
                    print(f"      Components found: {len(text_metadata.get('found_components', []))}")
                    print(f"      Generation parameters: {text_metadata.get('has_generation_params', False)}")
                    
                    # Show some details
                    for ref in text_metadata.get('model_references', [])[:2]:
                        print(f"      - Model ref: {ref['type']} = {ref['value']}")
                    
                except Exception as e:
                    print(f"   ❌ Error reading text file: {e}")
            else:
                print(f"   ℹ️ No associated text file found")
                
                # Create a sample text file for testing
                sample_text_path = image_path.replace('.png', '.txt').replace('.webp', '.txt')
                if not os.path.exists(sample_text_path):
                    sample_content = f"""Generation parameters for {os.path.basename(image_path)}

Steps: 25, Sampler: Euler a, CFG scale: 7.5, Seed: 123456789
Model: sample_model_v1.safetensors
Model hash: abc123def456
<lora:test_style:0.8>, <lora:detail_enhancer:0.5>
VAE: sample_vae.safetensors

Prompt: masterpiece, best quality, detailed face
Negative: (worst quality:1.4), (low quality:1.4)
"""
                    
                    print(f"   📝 Creating sample text file: {os.path.basename(sample_text_path)}")
                    with open(sample_text_path, 'w', encoding='utf-8') as f:
                        f.write(sample_content)
                    
                    # Now test reading it
                    try:
                        text_metadata = read_metadata_text_file(sample_text_path)
                        print(f"   ✅ Sample text file parsed:")
                        print(f"      Model references: {len(text_metadata.get('model_references', []))}")
                        print(f"      Components found: {len(text_metadata.get('found_components', []))}")
                    except Exception as e:
                        print(f"   ❌ Error with sample text file: {e}")

def test_civitai_cross_reference():
    """Test Civitai database cross-referencing"""
    print("\n\n🔗 Testing Civitai Cross-Reference Function")
    print("=" * 50)
    
    civitai_db_path = "Database/civitai.sqlite"
    
    if not os.path.exists(civitai_db_path):
        print(f"❌ Civitai database not found: {civitai_db_path}")
        print("   Cross-referencing tests skipped")
        return
    
    print(f"✅ Civitai database found: {civitai_db_path}")
    
    # Test with some sample hashes (you can replace with real ones)
    test_hashes = [
        ("abc123def456789", "sample_blur_hash_1"),
        ("def456ghi789012", "sample_blur_hash_2"),
        ("ghi789jkl012345", "")  # Test without blur hash
    ]
    
    for sha256_hash, blur_hash in test_hashes:
        print(f"\n🔍 Testing hash: {sha256_hash}")
        
        try:
            result = cross_reference_with_civitai(sha256_hash, blur_hash, civitai_db_path)
            
            print(f"   ✅ Cross-reference completed")
            print(f"   📊 Civitai matches: {len(result.get('civitai_matches', []))}")
            print(f"   🔗 Local matches: {len(result.get('local_matches', []))}")
            print(f"   ⚠️ Errors: {len(result.get('errors', []))}")
            
            # Show some match details
            for match in result.get('civitai_matches', [])[:2]:
                print(f"      - Match: {match.get('name', 'Unknown')} (type: {match.get('type', 'Unknown')})")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")

def test_database_functions():
    """Test database functions with the FileScanner"""
    print("\n\n🗃️ Testing Database Functions")
    print("=" * 50)
    
    # Initialize scanner
    config_file = "config.ini"
    if not os.path.exists(config_file):
        print("⚠️ Config file not found, using defaults")
        # Create a minimal config for testing
        with open(config_file, 'w') as f:
            f.write("""[DEFAULT]
source_directory = Lora_example
destination_directory = ./test_output
database_path = Database/file_scanner.sqlite
verbose = true
""")
    
    scanner = FileScanner(config_file)
    
    try:
        print(f"✅ Scanner initialized")
        
        # Test scanning the sample directories
        print(f"\n📂 Scanning sample directories...")
        sample_models = "sample_models"
        results = scanner.scan_directory(sample_models) if os.path.exists(sample_models) else {"models": [], "images": [], "text_files": [], "other_files": []}
        
        print(f"   📊 Scan results:")
        print(f"      Models: {len(results.get('models', []))}")
        print(f"      Images: {len(results.get('images', []))}")
        print(f"      Text files: {len(results.get('text_files', []))}")
        print(f"      Other files: {len(results.get('other_files', []))}")
        
        # Test database queries
        print(f"\n🗃️ Testing database methods...")
        
        # Test finding models by hash (using a sample hash)
        sample_hash = "abc123def456"
        model_result = scanner.db.find_model_by_hash(sample_hash)
        print(f"   🔍 Model by hash search: {'Found' if model_result else 'Not found'}")
        
        # Test component search
        component_results = scanner.db.find_component_by_name("test_component")
        print(f"   🧩 Component search: {len(component_results)} results")
        
        # Test orphaned media detection
        orphaned_files = scanner.get_orphaned_media_files()
        print(f"   🏃 Orphaned media files: {len(orphaned_files)}")
        
        if orphaned_files and len(orphaned_files) < 20:  # Only show if reasonable number
            print(f"      Sample orphaned files:")
            for orphaned in orphaned_files[:3]:
                print(f"      - {orphaned.get('file_name', 'Unknown')}")
        elif len(orphaned_files) >= 20:
            print(f"      ⚠️ Large number of orphaned files detected - likely database issue")
            print(f"      Showing first 3:")
            for orphaned in orphaned_files[:3]:
                print(f"      - {orphaned.get('file_name', 'Unknown')} (ID: {orphaned.get('id', 'N/A')})")
        
    except Exception as e:
        print(f"❌ Database test error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        scanner.close()

def test_file_organization():
    """Test the new file organization logic"""
    print("\n\n📁 Testing File Organization Logic")
    print("=" * 50)
    
    # Create a test config if needed
    config_file = "config.ini"
    if not os.path.exists(config_file):
        with open(config_file, 'w') as f:
            f.write("""[DEFAULT]
source_directory = Lora_example
destination_directory = ./test_output
database_path = Database/file_scanner.sqlite
verbose = true
use_model_type_subfolders = true
""")
    
    # Initialize scanner
    scanner = FileScanner("config.ini")
    
    try:
        # Test model directory generation
        sample_model_info = {
            'component_type': 'lora',
            'base_model': 'SD 1.5', 
            'component_name': 'test_style_lora',
            'file_name': 'sample_style_lora.safetensors'
        }
        
        model_dir = scanner._get_model_directory(sample_model_info)
        print(f"📂 Model directory: {model_dir}")
        
        # Test duplicate directory generation
        duplicate_dir = scanner._get_duplicate_media_directory(sample_model_info)
        print(f"📂 Duplicate directory: {duplicate_dir}")
        
        # Test with different model types
        model_types = ['checkpoint', 'lora', 'vae', 'embedding']
        for model_type in model_types:
            test_info = {
                'component_type': model_type,
                'base_model': 'SD 1.5',
                'component_name': f'test_{model_type}',
            }
            
            dir_path = scanner._get_model_directory(test_info)
            print(f"   {model_type}: {dir_path}")
        
    except Exception as e:
        print(f"❌ Organization test error: {e}")
    finally:
        scanner.close()

def main():
    """Run all tests"""
    print("🚀 Advanced Metadata and Cross-Reference Testing Suite")
    print("=" * 60)
    print("Testing all new functions with sample directories\n")
    
    try:
        # Run all test suites
        test_metadata_extraction()
        test_generation_parameter_parsing()
        test_metadata_text_files()
        test_civitai_cross_reference()
        test_database_functions()
        test_file_organization()
        
        print("\n\n🎉 Testing Complete!")
        print("=" * 60)
        print("✅ All test suites executed")
        print("📊 Check output above for detailed results")
        print("⚠️ Any errors indicate areas that need attention")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Testing interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Testing failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()