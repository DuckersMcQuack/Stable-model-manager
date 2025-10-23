#!/usr/bin/env python3
"""
Simple test script for incremental metadata scanning
"""

import sys
import os

# Add parent directory to path for importing main modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from file_scanner import FileScanner

def main():
    """Test incremental metadata scanning"""
    print("🔍 Testing Incremental Metadata System")
    print("=" * 50)
    
    scanner = None
    try:
        # Initialize scanner
        scanner = FileScanner("config.ini")
        
        # Test 1: List available metadata fields
        print("\n1️⃣ Available metadata fields:")
        fields = scanner.get_available_metadata_fields()
        for field_name, description in list(fields.items())[:5]:
            print(f"   📊 {field_name}: {description}")
        print(f"   ... and {len(fields) - 5} more fields")
        
        # Test 2: Check if we have any files in the database
        cursor = scanner.db.conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM scanned_files WHERE file_type = "image"')
        image_count = cursor.fetchone()[0]
        print(f"\n2️⃣ Database status:")
        print(f"   📁 Image files in database: {image_count}")
        
        if image_count == 0:
            print(f"   ⚠️ No image files found. Need to scan some files first.")
            return
        
        # Test 3: Get the first image file for testing
        cursor.execute('SELECT id, file_name FROM scanned_files WHERE file_type = "image" LIMIT 1')
        test_file = cursor.fetchone()
        
        if test_file:
            file_id, file_name = test_file
            print(f"   🎯 Testing with: {file_name} (ID: {file_id})")
            
            # Test 4: Run incremental metadata scan with a few fields
            test_checkboxes = {
                'width': True,
                'height': True,
                'steps': True,
                'sampler': False,
                'cfg_scale': False,
                'seed': True,
                'model_hash': False,
                # Set all other fields to False
                **{field: False for field in fields.keys() if field not in ['width', 'height', 'steps', 'seed']}
            }
            
            print(f"\n3️⃣ Running incremental scan for selected fields...")
            results = scanner.scan_metadata_incrementally(file_id, test_checkboxes)
            
            print(f"   ✅ Fields scanned: {len(results.get('scanned_fields', []))}")
            print(f"   ⏭️ Fields skipped: {len(results.get('skipped_fields', []))}")
            print(f"   ❌ Fields failed: {len(results.get('failed_fields', []))}")
            
            # Show some results
            if results.get('scanned_fields'):
                print(f"\n   🎯 Sample extracted data:")
                for result in results['scanned_fields'][:3]:
                    print(f"      📊 {result}")
        
        # Test 5: Generate a metadata scan report
        print(f"\n4️⃣ Metadata scan report summary:")
        report = scanner.get_metadata_scan_report()
        
        if report.get('results'):
            for result in report['results'][:5]:
                field_name = result.get('field_name', 'Unknown')
                successful = result.get('successful', 0)
                total = result.get('total_attempts', 0)
                success_rate = (successful / total * 100) if total > 0 else 0
                print(f"   📊 {field_name:15} {successful:>3}/{total:>3} ({success_rate:>5.1f}%)")
        
        print(f"\n✅ Incremental metadata system is working!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if scanner:
            scanner.close()

if __name__ == "__main__":
    main()