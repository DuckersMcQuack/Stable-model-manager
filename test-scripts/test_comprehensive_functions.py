#!/usr/bin/env python3
"""
Comprehensive test of all script functions on sample directories
Tests file scanning, metadata extraction, cross-referencing, and incremental scanning
"""

import os
import sys
import json
from pathlib import Path

# Add parent directory to path for importing main modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from file_scanner import FileScanner, DatabaseManager, extract_image_metadata, extract_comprehensive_metadata, cross_reference_with_civitai
from metadata_extractor import MetadataExtractor
from duplicate_detector import DuplicateDetector

def test_file_scanning():
    """Test basic file scanning functionality"""
    print("=" * 60)
    print("TESTING FILE SCANNING FUNCTIONALITY")
    print("=" * 60)
    
        # Initialize scanner with sample directories as source
    scanner = FileScanner("config.ini")
    scanner.config['source_directory'] = './sample_models'
    
    print(f"Scanning directory: {scanner.config['source_directory']}")
    
    # Scan the directory
    results = scanner.scan_directory('./sample_models')
    
    print(f"\nScan Results:")
    print(f"  Models found: {len(results['models'])}")
    print(f"  Images found: {len(results['images'])}")
    print(f"  Text files found: {len(results['text_files'])}")
    print(f"  Other files found: {len(results['other_files'])}")
    
    # Show some examples
    if results['models']:
        print(f"\nSample model files:")
        for model in results['models'][:3]:
            print(f"  - {model['file_name']} ({model['file_size']} bytes)")
    
    if results['images']:
        print(f"\nSample image files:")
        for image in results['images'][:3]:
            print(f"  - {image['file_name']} ({image['file_size']} bytes)")
    
    return results

def test_metadata_extraction():
    """Test metadata extraction on sample files"""
    print("\n" + "=" * 60)
    print("TESTING METADATA EXTRACTION")
    print("=" * 60)
    
    lora_dir = Path('./sample_media')
    
    # Test image metadata extraction
    image_files = list(lora_dir.glob('*.png')) + list(lora_dir.glob('*.webp')) + list(lora_dir.glob('*.jpg'))
    
    for image_file in image_files[:3]:  # Test first 3 images
        print(f"\nTesting image: {image_file.name}")
        
        # Test basic image metadata
        basic_metadata = extract_image_metadata(str(image_file))
        print(f"  Basic metadata extracted: {basic_metadata.get('error') is None}")
        if basic_metadata.get('width') and basic_metadata.get('height'):
            print(f"  Dimensions: {basic_metadata['width']}x{basic_metadata['height']}")
        
        # Test comprehensive metadata
        comp_metadata = extract_comprehensive_metadata(str(image_file), basic_metadata)
        print(f"  Civitai ID: {comp_metadata.get('civitai_id')}")
        print(f"  Has prompt: {comp_metadata.get('has_positive_prompt') == 1}")
        print(f"  Generation tool: {comp_metadata.get('generation_tool')}")
        
        if comp_metadata.get('components'):
            print(f"  Components found: {len(comp_metadata['components'])}")
            for comp in comp_metadata['components'][:2]:
                print(f"    - {comp['type']}: {comp['name']} (weight: {comp['weight']})")

def test_database_operations():
    """Test database operations and querying"""
    print("\n" + "=" * 60)
    print("TESTING DATABASE OPERATIONS")
    print("=" * 60)
    
    db = DatabaseManager()
    
    # Get some basic stats
    cursor = db.conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM scanned_files")
    total_files = cursor.fetchone()[0]
    print(f"Total files in database: {total_files}")
    
    cursor.execute("SELECT COUNT(*) FROM scanned_files WHERE file_type = 'image'")
    image_files = cursor.fetchone()[0]
    print(f"Image files: {image_files}")
    
    cursor.execute("SELECT COUNT(*) FROM scanned_files WHERE file_type = 'model'")
    model_files = cursor.fetchone()[0]
    print(f"Model files: {model_files}")
    
    cursor.execute("SELECT COUNT(*) FROM media_metadata")
    media_metadata = cursor.fetchone()[0]
    print(f"Media metadata records: {media_metadata}")
    
    # Test getting files needing metadata scan
    files_needing_scan = db.get_files_needing_metadata_scan(5)
    print(f"Files needing metadata scan: {len(files_needing_scan)}")
    
    if files_needing_scan:
        print("Sample files needing scan:")
        for file_info in files_needing_scan[:3]:
            print(f"  - {file_info['file_name']} (scan_status: {file_info['scan_status']})")
    
    db.close()

def test_cross_referencing():
    """Test cross-referencing functionality"""
    print("\n" + "=" * 60)
    print("TESTING CROSS-REFERENCING")
    print("=" * 60)
    
    # Test with sample media files 
    lora_dir = Path('./sample_media')
    image_files = list(lora_dir.glob('*.png')) + list(lora_dir.glob('*.webp'))
    
    for image_file in image_files[:2]:  # Test first 2 images
        print(f"\nTesting cross-reference for: {image_file.name}")
        
        # Calculate hash for cross-reference
        from file_scanner import FileScanner
        scanner = FileScanner()
        sha256_hash, _ = scanner.calculate_hashes(str(image_file))
        
        # Test cross-referencing (this will attempt to use civitai database if available)
        try:
            cross_ref_results = cross_reference_with_civitai(sha256_hash)
            print(f"  Cross-reference results:")
            print(f"    Found models: {len(cross_ref_results.get('found_models', []))}")
            print(f"    Found components: {len(cross_ref_results.get('found_components', []))}")
            print(f"    Civitai matches: {len(cross_ref_results.get('civitai_matches', []))}")
        except Exception as e:
            print(f"  Cross-reference failed (expected if no civitai.sqlite): {e}")

def test_incremental_metadata():
    """Test incremental metadata functionality"""
    print("\n" + "=" * 60)
    print("TESTING INCREMENTAL METADATA SYSTEM")
    print("=" * 60)
    
    db = DatabaseManager()
    scanner = FileScanner()
    
    # Test getting available metadata fields
    available_fields = scanner.get_available_metadata_fields()
    
    print(f"Available metadata fields: {len(available_fields)}")
    print("Sample fields:")
    field_names = list(available_fields.keys())
    for field in field_names[:5]:
        print(f"  - {field}: {available_fields[field]}")
    
    # Test metadata scan status methods
    cursor = db.conn.cursor()
    cursor.execute("SELECT id FROM scanned_files WHERE file_type = 'image' LIMIT 1")
    result = cursor.fetchone()
    
    if result:
        file_id = result[0]
        print(f"\nTesting with file ID: {file_id}")
        
        # Test metadata scan status table
        cursor.execute("SELECT COUNT(*) FROM metadata_scan_status WHERE scanned_file_id = ?", (file_id,))
        scan_count = cursor.fetchone()[0]
        print(f"Metadata scan records for file: {scan_count}")
        
        # Test available metadata functionality
        print(f"Available metadata fields can be extracted: {len(field_names)} fields")
    
    db.close()

def test_duplicate_detection():
    """Test duplicate detection functionality"""
    print("\n" + "=" * 60)
    print("TESTING DUPLICATE DETECTION")
    print("=" * 60)
    
    try:
        detector = DuplicateDetector()
        
        # Get duplicate stats using the correct method name
        stats = detector.get_duplicate_summary()
        print(f"Duplicate detection stats:")
        print(f"  Total duplicates: {stats.get('total_duplicates', 0)}")
        print(f"  Duplicate groups: {stats.get('duplicate_groups', 0)}")
        print(f"  AutoV3 duplicates: {stats.get('autov3_duplicates', 0)}")
        
    except Exception as e:
        print(f"Duplicate detection test failed: {e}")

def test_metadata_extractor_cli():
    """Test the CLI metadata extractor"""
    print("\n" + "=" * 60)
    print("TESTING METADATA EXTRACTOR CLI")
    print("=" * 60)
    
    try:
        extractor = MetadataExtractor()
        
        # Get some file IDs to test with
        db = DatabaseManager()
        cursor = db.conn.cursor()
        cursor.execute("SELECT id, file_name, file_path FROM scanned_files WHERE file_type = 'model' LIMIT 3")
        sample_files = cursor.fetchall()
        
        for file_id, file_name, file_path in sample_files:
            print(f"\nTesting file ID {file_id}: {file_name}")
            try:
                # Check if file exists before processing
                if os.path.exists(file_path):
                    # Test creating comprehensive metadata
                    metadata = extractor.create_comprehensive_metadata(file_path)
                    print(f"  Metadata created successfully: {len(str(metadata))} characters")
                    print(f"  Base model: {metadata.get('base_model', 'Unknown')}")
                    print(f"  Model type: {metadata.get('model_type', 'Unknown')}")
                else:
                    print(f"  File does not exist: {file_path}")
            except Exception as e:
                print(f"  Metadata creation failed: {e}")
        
        db.close()
        
    except Exception as e:
        print(f"Metadata extractor CLI test failed: {e}")

def main():
    """Run all comprehensive tests"""
    print("COMPREHENSIVE FUNCTION TESTING")
    print("Testing all script functions on sample directories")
    print("=" * 80)
    
    try:
        # Test 1: File Scanning
        scan_results = test_file_scanning()
        
        # Test 2: Metadata Extraction
        test_metadata_extraction()
        
        # Test 3: Database Operations
        test_database_operations()
        
        # Test 4: Cross-referencing
        test_cross_referencing()
        
        # Test 5: Incremental Metadata
        test_incremental_metadata()
        
        # Test 6: Duplicate Detection
        test_duplicate_detection()
        
        # Test 7: Metadata Extractor CLI
        test_metadata_extractor_cli()
        
        print("\n" + "=" * 80)
        print("COMPREHENSIVE TESTING COMPLETE")
        print("All major functions have been tested!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nTest suite failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()