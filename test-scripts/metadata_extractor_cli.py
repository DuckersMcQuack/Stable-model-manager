#!/usr/bin/env python3
"""
Interactive metadata extraction tool with checkbox selection
Demonstrates incremental metadata scanning that only adds missing fields
"""

import os
import argparse
from file_scanner import FileScanner

def main():
    """Interactive metadata extraction with selective field scanning"""
    parser = argparse.ArgumentParser(description="Incremental Metadata Scanner with Checkbox Selection")
    parser.add_argument("--config", default="config.ini", help="Configuration file path")
    parser.add_argument("--file-id", type=int, help="Specific file ID to process")
    parser.add_argument("--file-type", default="image", help="File type to process (image, video)")
    parser.add_argument("--preset", choices=['basic', 'full', 'generation-only', 'civitai-only'], 
                       help="Preset field selections")
    parser.add_argument("--list-files", action="store_true", help="List available files in database")
    parser.add_argument("--show-report", action="store_true", help="Show metadata scan status report")
    
    args = parser.parse_args()
    
    # Initialize scanner
    scanner = FileScanner(args.config)
    
    try:
        if args.list_files:
            list_available_files(scanner, args.file_type)
            return
        
        if args.show_report:
            show_metadata_report(scanner, args.file_id)
            return
        
        # Define field presets
        field_presets = {
            'basic': ['width', 'height', 'steps', 'sampler', 'cfg_scale', 'seed', 'model_hash'],
            'full': list(scanner.get_available_metadata_fields().keys()),
            'generation-only': ['steps', 'sampler', 'cfg_scale', 'seed', 'model_hash', 'generation_tool', 'prompt_text', 'negative_prompt'],
            'civitai-only': ['civitai_id', 'civitai_uuid', 'blur_hash', 'nsfw_level']
        }
        
        # Determine which fields to scan
        if args.preset:
            selected_fields = field_presets[args.preset]
            checkboxes = {field: field in selected_fields for field in scanner.get_available_metadata_fields().keys()}
            print(f"📋 Using preset: {args.preset} ({len(selected_fields)} fields)")
        else:
            checkboxes = interactive_field_selection(scanner)
        
        # Process files
        if args.file_id:
            process_single_file(scanner, args.file_id, checkboxes)
        else:
            process_multiple_files(scanner, args.file_type, checkboxes)
        
    except KeyboardInterrupt:
        print("\n⚠️ Operation cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        scanner.close()

def list_available_files(scanner: FileScanner, file_type: str):
    """List files available for metadata extraction"""
    cursor = scanner.db.conn.cursor()
    cursor.execute('''
        SELECT sf.id, sf.file_name, sf.file_type, sf.file_path,
               COUNT(mss.field_name) as scanned_fields
        FROM scanned_files sf
        LEFT JOIN metadata_scan_status mss ON sf.id = mss.scanned_file_id AND mss.scan_status = 1
        WHERE sf.file_type = ?
        GROUP BY sf.id
        ORDER BY sf.file_name
        LIMIT 20
    ''', (file_type,))
    
    files = cursor.fetchall()
    
    print(f"📁 Available {file_type} files in database:")
    print("=" * 80)
    print(f"{'ID':>4} {'Scanned':>7} {'File Name':<40} {'Path'}")
    print("-" * 80)
    
    for file_id, file_name, file_type, file_path, scanned_fields in files:
        path_short = file_path[-35:] if len(file_path) > 35 else file_path
        print(f"{file_id:>4} {scanned_fields:>7} {file_name[:40]:<40} {path_short}")
    
    if len(files) == 20:
        print("... (showing first 20 files)")

def interactive_field_selection(scanner: FileScanner) -> dict:
    """Interactive selection of metadata fields"""
    fields = scanner.get_available_metadata_fields()
    
    print("\n🎯 Select Metadata Fields to Extract:")
    print("=" * 60)
    
    # Show fields in categories
    categories = {
        'Basic Properties': ['width', 'height'],
        'Generation Parameters': ['steps', 'sampler', 'cfg_scale', 'seed', 'denoising_strength', 'clip_skip'],
        'Model Information': ['model_name', 'model_hash', 'vae_name', 'vae_hash', 'generation_tool'],
        'Prompts': ['prompt_text', 'negative_prompt'],
        'Upscaling': ['hires_upscaler', 'hires_steps', 'hires_upscale'],
        'Components': ['has_components', 'component_count'],
        'Civitai Data': ['civitai_id', 'civitai_uuid', 'blur_hash', 'nsfw_level']
    }
    
    selected_fields = []
    
    for category, field_list in categories.items():
        print(f"\n📂 {category}:")
        for field in field_list:
            if field in fields:
                description = fields[field]
                include = input(f"   ☐ {field:18} - {description} (y/n): ").strip().lower()
                if include in ['y', 'yes', '1']:
                    selected_fields.append(field)
    
    return {field: field in selected_fields for field in fields.keys()}

def process_single_file(scanner: FileScanner, file_id: int, checkboxes: dict):
    """Process a single file"""
    cursor = scanner.db.conn.cursor()
    cursor.execute('SELECT file_name, file_path FROM scanned_files WHERE id = ?', (file_id,))
    file_info = cursor.fetchone()
    
    if not file_info:
        print(f"❌ File ID {file_id} not found")
        return
    
    file_name, file_path = file_info
    print(f"\n🔍 Processing: {file_name}")
    
    results = scanner.scan_metadata_incrementally(file_id, checkboxes)
    
    print_scan_results(results)

def process_multiple_files(scanner: FileScanner, file_type: str, checkboxes: dict):
    """Process multiple files of the specified type"""
    cursor = scanner.db.conn.cursor()
    cursor.execute('''
        SELECT id, file_name FROM scanned_files 
        WHERE file_type = ? 
        ORDER BY file_name 
        LIMIT 10
    ''', (file_type,))
    
    files = cursor.fetchall()
    
    if not files:
        print(f"❌ No {file_type} files found in database")
        return
    
    print(f"\n🔍 Processing {len(files)} {file_type} files...")
    
    total_stats = {'scanned': 0, 'skipped': 0, 'failed': 0}
    
    for file_id, file_name in files:
        print(f"\n📁 {file_name}")
        
        results = scanner.scan_metadata_incrementally(file_id, checkboxes)
        
        # Quick summary
        scanned = len(results['scanned_fields'])
        skipped = len(results['skipped_fields'])
        failed = len(results['failed_fields'])
        
        total_stats['scanned'] += scanned
        total_stats['skipped'] += skipped
        total_stats['failed'] += failed
        
        print(f"   ✅ {scanned} scanned, ⏭️ {skipped} skipped, ❌ {failed} failed")
        
        # Show any newly extracted data
        if results['scanned_fields'][:2]:
            for field_result in results['scanned_fields'][:2]:
                print(f"      📊 {field_result}")
    
    print(f"\n📊 Total Summary:")
    print(f"   ✅ Fields extracted: {total_stats['scanned']}")
    print(f"   ⏭️ Fields skipped: {total_stats['skipped']}")
    print(f"   ❌ Fields failed: {total_stats['failed']}")

def show_metadata_report(scanner: FileScanner, file_id: int | None = None):
    """Show metadata scanning report"""
    report = scanner.get_metadata_scan_report(file_id)
    
    if report['report_type'] == 'file_specific':
        print(f"\n📊 Metadata Report for File ID {file_id}:")
        print("=" * 60)
        
        if report['results']:
            for result in report['results']:
                field_name = result.get('field_name', 'Unknown')
                scan_status = result.get('scan_status', 0)
                field_value = result.get('field_value', 'None')
                scan_notes = result.get('scan_notes', '')
                
                status_text = {0: '⚪ Not scanned', 1: '✅ Success', 2: '❌ Failed', 3: '⚫ N/A'}
                print(f"  {field_name:18} {status_text.get(scan_status, '❓ Unknown')}: {field_value}")
                
                if scan_notes:
                    print(f"                     Notes: {scan_notes}")
    else:
        print(f"\n📊 Metadata Scanning Summary Report:")
        print("=" * 80)
        print(f"{'Field Name':18} {'Success':>7} {'Failed':>6} {'N/A':>5} {'Total':>6} {'Rate':>6}")
        print("-" * 80)
        
        for result in report['results']:
            field_name = result['field_name']
            successful = result['successful'] or 0
            failed = result['failed'] or 0
            not_applicable = result['not_applicable'] or 0
            total = result['total_attempts'] or 0
            
            success_rate = (successful / total * 100) if total > 0 else 0
            print(f"{field_name:18} {successful:>7} {failed:>6} {not_applicable:>5} {total:>6} {success_rate:>5.1f}%")

def print_scan_results(results: dict):
    """Print detailed scan results"""
    if 'error' in results:
        print(f"❌ Error: {results['error']}")
        return
    
    print(f"\n📊 Scan Results:")
    print(f"   ✅ Fields scanned: {len(results['scanned_fields'])}")
    print(f"   ⏭️ Fields skipped: {len(results['skipped_fields'])}")  
    print(f"   ❌ Fields failed: {len(results['failed_fields'])}")
    
    if results['scanned_fields']:
        print(f"\n🎯 Newly Extracted Metadata:")
        for field_result in results['scanned_fields']:
            print(f"      ✅ {field_result}")
    
    if results['failed_fields']:
        print(f"\n❌ Failed Fields:")
        for field_error in results['failed_fields']:
            print(f"      ❌ {field_error}")
    
    if results['skipped_fields']:
        print(f"\n⏭️ Skipped Fields:")
        for skip_reason in results['skipped_fields'][:5]:
            print(f"      ⏭️ {skip_reason}")
        if len(results['skipped_fields']) > 5:
            print(f"      ... and {len(results['skipped_fields']) - 5} more")

if __name__ == "__main__":
    main()