#!/usr/bin/env python3
"""
Demo script to showcase the advanced cross-referencing capabilities
"""

import os
import sys
from file_scanner import FileScanner, cross_reference_with_civitai

def demo_cross_referencing():
    """Demonstrate the cross-referencing capabilities"""
    print("🚀 Advanced Stable Diffusion Model Cross-Referencing Demo")
    print("=" * 60)
    
    # Initialize scanner
    config_file = "config.ini"
    if not os.path.exists(config_file):
        print(f"❌ Configuration file '{config_file}' not found.")
        print("Please ensure config.ini exists in the current directory.")
        return 1
    
    scanner = FileScanner(config_file)
    
    try:
        print(f"\n📊 Database Overview:")
        
        # Show database stats
        cursor = scanner.db.conn.cursor()
        
        # Count scanned files
        cursor.execute("SELECT COUNT(*) FROM scanned_files")
        total_files = cursor.fetchone()[0]
        
        # Count by file type
        cursor.execute("SELECT file_type, COUNT(*) FROM scanned_files GROUP BY file_type")
        file_types = dict(cursor.fetchall())
        
        print(f"   Total files in database: {total_files}")
        for file_type, count in file_types.items():
            print(f"   {file_type}: {count}")
        
        # Show media metadata stats
        cursor.execute("SELECT COUNT(*) FROM media_metadata")
        metadata_count = cursor.fetchone()[0]
        print(f"   Files with metadata: {metadata_count}")
        
        # Show component usage stats
        cursor.execute("SELECT COUNT(*) FROM component_usage")
        component_usage_count = cursor.fetchone()[0]
        print(f"   Component usage records: {component_usage_count}")
        
        if component_usage_count > 0:
            print(f"\n🎯 Component Usage Examples:")
            cursor.execute('''
                SELECT component_type, component_name, component_weight, usage_context
                FROM component_usage
                LIMIT 5
            ''')
            
            for comp_type, comp_name, weight, context in cursor.fetchall():
                print(f"   {comp_type}: {comp_name} (weight: {weight}) - {context}")
        
        # Show cross-reference capabilities
        print(f"\n🔗 Cross-Reference Capabilities:")
        
        # Test civitai cross-referencing
        civitai_db = "Database/civitai.sqlite"
        if os.path.exists(civitai_db):
            print(f"   ✅ Civitai database found: {civitai_db}")
            
            # Show some sample data
            import sqlite3
            civitai_conn = sqlite3.connect(civitai_db)
            civitai_cursor = civitai_conn.cursor()
            
            # Count models in civitai db
            try:
                civitai_cursor.execute("SELECT COUNT(*) FROM models")
                civitai_models = civitai_cursor.fetchone()[0]
                print(f"   📚 Civitai models available: {civitai_models}")
            except:
                print(f"   📚 Civitai database structure unknown - will attempt cross-referencing")
            
            civitai_conn.close()
        else:
            print(f"   ⚠️  Civitai database not found: {civitai_db}")
            print(f"      Cross-referencing will be limited to local data")
        
        # Show advanced features
        print(f"\n🔧 Advanced Processing Features:")
        print(f"   ✅ Multi-hash system (SHA256 + AutoV3 + BlurHash)")
        print(f"   ✅ Comprehensive metadata extraction (PNG parameters, EXIF)")
        print(f"   ✅ Component detection (LoRA, LyCO, VAE, ControlNet)")
        print(f"   ✅ Generation tool detection (Automatic1111, ComfyUI)")
        print(f"   ✅ Proximity-based sorting (media next to models)")
        print(f"   ✅ Text metadata file parsing")
        print(f"   ✅ Duplicate categorization system")
        print(f"   ✅ Configurable folder structure")
        
        print(f"\n📋 Usage Examples:")
        print(f"   # Scan a new directory")
        print(f"   python file_scanner.py --scan-directory /path/to/models --verbose")
        print(f"")
        print(f"   # Process orphaned media files with cross-referencing")
        print(f"   python file_scanner.py --process-orphaned --verbose")
        print(f"")
        print(f"   # Extract metadata from media files")
        print(f"   python file_scanner.py --extract-metadata --verbose")
        print(f"")
        print(f"   # Dry run to see what would happen")
        print(f"   python file_scanner.py --process-orphaned --dry-run")
        
        print(f"\n🎯 Cross-Referencing Logic:")
        print(f"   1. Media file found → Check embedded metadata for model hash")
        print(f"   2. No embedded metadata → Look for .txt metadata file")
        print(f"   3. Parse text file → Extract LoRA/checkpoint references")
        print(f"   4. No text file → Use SHA256+BlurHash to query Civitai")
        print(f"   5. Match found → Move to model directory")
        print(f"   6. Duplicate exists → Move to duplicates/media/[type]/")
        print(f"   7. No match found → Mark as ignored")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        scanner.close()

if __name__ == "__main__":
    exit(demo_cross_referencing())