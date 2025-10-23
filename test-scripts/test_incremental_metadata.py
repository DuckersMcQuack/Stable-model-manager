#!/usr/bin/env python3
"""
Test script for incremental metadata scanning with checkbox interface
Demonstrates how the new system only adds missing metadata fields
"""

import os
import sys
from typing import Dict, List

# Add parent directory to path for importing main modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from file_scanner import FileScanner

def display_available_fields(scanner: FileScanner):
    """Display all available metadata fields with descriptions"""
    print("📋 Available Metadata Fields:")
    print("=" * 60)
    
    fields = scanner.get_available_metadata_fields()
    for field_name, description in fields.items():
        print(f"  ☐ {field_name:18} - {description}")
    
    return list(fields.keys())

def get_user_checkboxes(available_fields: List[str]) -> Dict[str, bool]:
    """Get user selections for which metadata fields to extract"""
    print(f"\n🎯 Select Metadata Fields to Extract:")
    print("=" * 60)
    print("Enter field numbers separated by spaces (e.g., 1 3 5 7)")
    print("Or 'all' for all fields, 'basic' for essential fields, 'none' to skip")
    
    # Display numbered options
    for i, field in enumerate(available_fields, 1):
        print(f"  {i:2d}. {field}")
    
    while True:
        try:
            user_input = input(f"\nYour selection: ").strip().lower()
            
            if user_input == 'all':
                return {field: True for field in available_fields}
            elif user_input == 'basic':
                basic_fields = ['width', 'height', 'steps', 'sampler', 'cfg_scale', 'seed', 'model_hash', 'generation_tool']
                return {field: field in basic_fields for field in available_fields}
            elif user_input == 'none':
                return {field: False for field in available_fields}
            else:
                # Parse numbers
                selected_numbers = [int(x.strip()) for x in user_input.split() if x.strip().isdigit()]
                selected_fields = [available_fields[i-1] for i in selected_numbers if 1 <= i <= len(available_fields)]
                return {field: field in selected_fields for field in available_fields}
                
        except (ValueError, IndexError) as e:
            print(f"❌ Invalid input. Please try again. Error: {e}")

def test_incremental_metadata_scanning():
    """Test the incremental metadata scanning system"""
    print("🔍 Incremental Metadata Scanning Test")
    print("=" * 60)
    
    # Initialize scanner
    config_file = "config.ini"
    if not os.path.exists(config_file):
        print("⚠️ Config file not found, using defaults")
        with open(config_file, 'w') as f:
            f.write("""[DEFAULT]
source_directory = Lora_example
destination_directory = ./test_output
database_path = Database/file_scanner.sqlite
verbose = true
""")
    
    scanner = FileScanner(config_file)
    
    try:
        # Get some sample files from database
        cursor = scanner.db.conn.cursor()
        cursor.execute('''
            SELECT id, file_name, file_type, file_path 
            FROM scanned_files 
            WHERE file_type = 'image' 
            LIMIT 5
        ''')
        
        image_files = cursor.fetchall()
        
        if not image_files:
            print("❌ No image files found in database. Please run a scan first.")
            print("Try: python file_scanner.py --scan-directory sample_models")
            return
        
        print(f"✅ Found {len(image_files)} image files in database")
        
        # Display available metadata fields
        available_fields = display_available_fields(scanner)
        
        # Get user selections
        metadata_checkboxes = get_user_checkboxes(available_fields)
        
        selected_count = sum(metadata_checkboxes.values())
        if selected_count == 0:
            print("\n⚠️ No fields selected. Exiting.")
            return
        
        print(f"\n🎯 Processing {selected_count} metadata fields for {len(image_files)} files")
        print("=" * 60)
        
        # Process each file
        for file_id, file_name, file_type, file_path in image_files:
            print(f"\n📁 Processing: {file_name}")
            
            # Check existing metadata status
            cursor.execute('''
                SELECT field_name, scan_status, field_value 
                FROM metadata_scan_status 
                WHERE scanned_file_id = ?
            ''', (file_id,))
            
            existing_metadata = {row[0]: {'status': row[1], 'value': row[2]} 
                               for row in cursor.fetchall()}
            
            if existing_metadata:
                print(f"   📊 Existing metadata: {len(existing_metadata)} fields already scanned")
                for field_name, info in list(existing_metadata.items())[:3]:
                    status_text = {0: 'Not scanned', 1: 'Success', 2: 'Failed', 3: 'N/A'}
                    print(f"      - {field_name}: {status_text.get(info['status'], 'Unknown')} = {info['value'] or 'None'}")
                if len(existing_metadata) > 3:
                    print(f"      ... and {len(existing_metadata) - 3} more")
            
            # Perform incremental scan
            results = scanner.scan_metadata_incrementally(file_id, metadata_checkboxes)
            
            # Display results
            if 'error' in results:
                print(f"   ❌ Error: {results['error']}")
                continue
            
            print(f"   ✅ Scan complete:")
            print(f"      📊 Fields scanned: {len(results['scanned_fields'])}")
            print(f"      ⏭️ Fields skipped: {len(results['skipped_fields'])}")  
            print(f"      ❌ Fields failed: {len(results['failed_fields'])}")
            
            # Show newly extracted metadata
            if results['scanned_fields']:
                print(f"      🎯 Newly extracted:")
                for field_result in results['scanned_fields'][:5]:
                    print(f"         - {field_result}")
                if len(results['scanned_fields']) > 5:
                    print(f"         ... and {len(results['scanned_fields']) - 5} more")
            
            # Show skipped fields
            if results['skipped_fields']:
                print(f"      ⏭️ Skipped:")
                for skip_reason in results['skipped_fields'][:3]:
                    print(f"         - {skip_reason}")
                if len(results['skipped_fields']) > 3:
                    print(f"         ... and {len(results['skipped_fields']) - 3} more")
        
        # Generate summary report
        print(f"\n📊 Summary Report:")
        print("=" * 60)
        
        report = scanner.get_metadata_scan_report()
        
        if report['results']:
            print(f"Metadata field scan statistics:")
            for field_stats in report['results']:
                field_name = field_stats['field_name']
                successful = field_stats['successful'] or 0
                failed = field_stats['failed'] or 0
                not_applicable = field_stats['not_applicable'] or 0
                total = field_stats['total_attempts'] or 0
                
                success_rate = (successful / total * 100) if total > 0 else 0
                print(f"  {field_name:18} - {successful:3d} success, {failed:3d} failed, {not_applicable:3d} N/A ({success_rate:5.1f}% success rate)")
        
        print(f"\n✅ Incremental metadata scanning test completed!")
        print(f"Database now contains selective metadata based on your checkbox selections.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        scanner.close()

def demo_checkbox_workflow():
    """Demonstrate the checkbox-based workflow"""
    print("🎯 Checkbox-Based Metadata Extraction Demo")
    print("=" * 60)
    print()
    print("This system allows you to:")
    print("✅ Select which metadata fields to extract (checkbox style)")
    print("✅ Skip already-scanned fields automatically")
    print("✅ Add new metadata fields to existing database entries")
    print("✅ Track scan status per field (success/failed/not applicable)")
    print("✅ Generate reports on metadata extraction coverage")
    print()
    print("Key Benefits:")
    print("📊 Incremental scanning - only scan what's needed")
    print("🚀 Efficient processing - skip duplicate work")
    print("🎛️ Flexible selection - choose what metadata you want")
    print("📈 Progress tracking - see what's been processed")
    print("🔄 Resumable scans - continue where you left off")
    print()
    
    proceed = input("Ready to test? (y/n): ").strip().lower()
    if proceed in ['y', 'yes']:
        test_incremental_metadata_scanning()
    else:
        print("Test cancelled.")

if __name__ == "__main__":
    demo_checkbox_workflow()