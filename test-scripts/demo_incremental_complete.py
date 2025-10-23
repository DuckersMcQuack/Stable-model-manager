#!/usr/bin/env python3
"""
Complete demonstration of incremental metadata system
Tests with actual LoRA example files
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from file_scanner import FileScanner

def main():
    """Comprehensive test of incremental metadata with real LoRA files"""
    print("🚀 Complete Incremental Metadata Demo")
    print("Testing with actual LoRA example files")
    print("=" * 60)
    
    scanner = FileScanner("config.ini")
    
    try:
        # Step 1: Get LoRA example files
        cursor = scanner.db.conn.cursor()
        cursor.execute('''
            SELECT id, file_name, file_path 
            FROM scanned_files 
            WHERE file_type = 'image' AND file_path LIKE '%Lora_example%'
            LIMIT 3
        ''')
        test_files = cursor.fetchall()
        
        print(f"\n📁 Found {len(test_files)} LoRA image files to test:")
        for file_id, name, path in test_files:
            print(f"   🖼️  {name} (ID: {file_id})")
        
        # Step 2: Show all available metadata fields
        fields = scanner.get_available_metadata_fields()
        print(f"\n📊 Available metadata fields ({len(fields)} total):")
        
        categories = {
            '🎨 Basic': ['width', 'height'],
            '⚙️ Generation': ['steps', 'sampler', 'cfg_scale', 'seed', 'denoising_strength'],
            '🤖 Model Info': ['model_name', 'model_hash', 'vae_name', 'generation_tool'],
            '💬 Prompts': ['prompt_text', 'negative_prompt'],
            '🔍 Civitai': ['civitai_id', 'civitai_uuid', 'nsfw_level']
        }
        
        for category, field_list in categories.items():
            print(f"   {category}: {', '.join(f for f in field_list if f in fields)}")
        
        # Step 3: Test incremental scanning with different scenarios
        print(f"\n🧪 Testing Incremental Metadata Scenarios")
        print("=" * 50)
        
        test_file_id = test_files[0][0]
        test_file_name = test_files[0][1]
        
        # Scenario 1: Basic image properties only
        print(f"\n1️⃣ Scenario: Basic image properties only")
        basic_checkboxes = {field: field in ['width', 'height'] for field in fields.keys()}
        
        results1 = scanner.scan_metadata_incrementally(test_file_id, basic_checkboxes)
        print(f"   ✅ Scanned: {len(results1.get('scanned_fields', []))}")
        print(f"   ⏭️ Skipped: {len(results1.get('skipped_fields', []))}")
        
        for result in results1.get('scanned_fields', []):
            print(f"      📊 {result}")
        
        # Scenario 2: Add generation parameters
        print(f"\n2️⃣ Scenario: Add generation parameters")
        generation_checkboxes = {field: field in ['width', 'height', 'steps', 'sampler', 'cfg_scale', 'seed'] for field in fields.keys()}
        
        results2 = scanner.scan_metadata_incrementally(test_file_id, generation_checkboxes)
        print(f"   ✅ Scanned: {len(results2.get('scanned_fields', []))}")
        print(f"   ⏭️ Skipped: {len(results2.get('skipped_fields', []))}")
        
        # Show newly scanned fields
        new_fields = [r for r in results2.get('scanned_fields', []) if 'width:' not in r and 'height:' not in r]
        for result in new_fields[:3]:
            print(f"      🆕 {result}")
        
        # Scenario 3: Add everything else 
        print(f"\n3️⃣ Scenario: Extract all remaining metadata")
        all_checkboxes = {field: True for field in fields.keys()}
        
        results3 = scanner.scan_metadata_incrementally(test_file_id, all_checkboxes)
        print(f"   ✅ Scanned: {len(results3.get('scanned_fields', []))}")
        print(f"   ⏭️ Skipped: {len(results3.get('skipped_fields', []))}")
        
        # Show newly scanned fields
        new_fields = results3.get('scanned_fields', [])
        if new_fields:
            print(f"   🆕 Additional fields extracted:")
            for result in new_fields[:5]:
                print(f"      📊 {result}")
            if len(new_fields) > 5:
                print(f"      ... and {len(new_fields) - 5} more")
        
        # Step 4: Show overall scan status
        print(f"\n📋 Final Scan Status Report")
        print("=" * 40)
        
        report = scanner.get_metadata_scan_report(test_file_id)
        if report.get('results'):
            successful_fields = []
            failed_fields = []
            na_fields = []
            
            for result in report['results']:
                field_name = result.get('field_name', 'Unknown')
                scan_status = result.get('scan_status', 0)
                field_value = result.get('field_value', 'None')
                
                if scan_status == 1:  # Success
                    successful_fields.append((field_name, field_value))
                elif scan_status == 2:  # Failed
                    failed_fields.append(field_name)
                elif scan_status == 3:  # N/A
                    na_fields.append(field_name)
            
            print(f"✅ Successfully extracted ({len(successful_fields)} fields):")
            for field_name, value in successful_fields[:8]:
                display_value = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
                print(f"   📊 {field_name:15}: {display_value}")
            
            if len(successful_fields) > 8:
                print(f"   ... and {len(successful_fields) - 8} more successful extractions")
            
            if failed_fields:
                print(f"\n❌ Failed fields ({len(failed_fields)}):")
                print(f"   {', '.join(failed_fields[:10])}")
            
            if na_fields:
                print(f"\n⚫ Not applicable ({len(na_fields)}):")
                print(f"   {', '.join(na_fields[:10])}")
        
        # Step 5: Test batch processing
        if len(test_files) > 1:
            print(f"\n🔄 Batch Processing Test")
            print("=" * 30)
            
            # Test selective extraction on multiple files
            selective_checkboxes = {field: field in ['width', 'height', 'steps', 'prompt_text'] for field in fields.keys()}
            
            total_scanned = 0
            total_skipped = 0
            
            for file_id, file_name, _ in test_files[1:3]:  # Test on 2 more files
                print(f"\n📁 Processing: {file_name}")
                results = scanner.scan_metadata_incrementally(file_id, selective_checkboxes)
                
                scanned = len(results.get('scanned_fields', []))
                skipped = len(results.get('skipped_fields', []))
                
                total_scanned += scanned
                total_skipped += skipped
                
                print(f"   ✅ {scanned} scanned, ⏭️ {skipped} skipped")
                
                # Show sample of extracted data
                for result in results.get('scanned_fields', [])[:2]:
                    print(f"      📊 {result}")
            
            print(f"\n📊 Batch Summary: {total_scanned} total extracted, {total_skipped} total skipped")
        
        print(f"\n🎉 Incremental Metadata Demo Complete!")
        print(f"\n✨ Key Features Demonstrated:")
        print(f"   ✅ Selective field extraction with checkboxes")
        print(f"   ✅ Automatic skip of already-extracted fields")
        print(f"   ✅ Progress tracking per field")
        print(f"   ✅ Batch processing capabilities")
        print(f"   ✅ Detailed status reporting")
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        scanner.close()

if __name__ == "__main__":
    main()