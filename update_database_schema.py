#!/usr/bin/env python3
"""
Database Schema Update Script
Adds missing columns to media_metadata table for enhanced metadata extraction.
"""

import sqlite3
import sys
import os

def update_media_metadata_schema(db_path):
    """Add missing columns to media_metadata table"""
    
    # List of columns that need to be added
    new_columns = [
        # Batch and processing parameters  
        ('batch_size', 'INTEGER'),
        ('batch_pos', 'INTEGER'),
        ('eta', 'REAL'),
        ('ensd', 'INTEGER'),
        
        # Face restoration
        ('face_restoration', 'TEXT'),
        ('restore_faces', 'INTEGER'),
        
        # Tiling and resolution
        ('tiled_diffusion', 'TEXT'),
        ('hires_resize_mode', 'TEXT'),
        ('first_pass_size', 'TEXT'),
        
        # AddNet/LoRA parameters
        ('addnet_enabled', 'INTEGER'),
        ('addnet_module_1', 'TEXT'),
        ('addnet_model_1', 'TEXT'),
        ('addnet_weight_1', 'REAL'),
    ]
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        print(f"Updating database schema: {db_path}")
        
        # Get existing columns
        cursor.execute("PRAGMA table_info(media_metadata)")
        existing_columns = {row[1] for row in cursor.fetchall()}
        
        added_columns = 0
        for column_name, column_type in new_columns:
            if column_name not in existing_columns:
                try:
                    sql = f"ALTER TABLE media_metadata ADD COLUMN {column_name} {column_type}"
                    cursor.execute(sql)
                    print(f"✅ Added column: {column_name} ({column_type})")
                    added_columns += 1
                except sqlite3.OperationalError as e:
                    print(f"❌ Failed to add column {column_name}: {e}")
            else:
                print(f"⚠️  Column {column_name} already exists")
        
        conn.commit()
        conn.close()
        
        print(f"\n🎉 Schema update complete: {added_columns} columns added")
        return True
        
    except Exception as e:
        print(f"❌ Database schema update failed: {e}")
        return False

def main():
    import configparser
    
    # Try to get database path from command line first, then config.ini
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    else:
        # Load from config.ini as fallback
        config = configparser.ConfigParser()
        config.read('config.ini')
        db_path = config.get('Database', 'path', fallback="model_sorter.sqlite")
        print(f"No database path provided, using config.ini: {db_path}")
    
    if not os.path.exists(db_path):
        print(f"❌ Database file not found: {db_path}")
        print("Usage: python update_database_schema.py <path_to_database>")
        print("Example: python update_database_schema.py model_sorter.sqlite")
        sys.exit(1)
    
    # Create backup
    backup_path = f"{db_path}.backup_{int(__import__('time').time())}"
    try:
        import shutil
        shutil.copy2(db_path, backup_path)
        print(f"📋 Created backup: {backup_path}")
    except Exception as e:
        print(f"⚠️  Could not create backup: {e}")
        response = input("Continue without backup? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Update schema
    success = update_media_metadata_schema(db_path)
    
    if success:
        print(f"\n✅ Database schema updated successfully!")
        print(f"   To run this script on your server: python update_database_schema.py <your_database_path>")
    else:
        print(f"\n❌ Schema update failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()