#!/usr/bin/env python3
"""
Add missing .metadata.json files to the scanned_files database
This is useful when metadata files were created after the initial scan
"""

import os
import sys
import sqlite3
import time
from pathlib import Path

def add_missing_metadata_files(db_path: str = "model_sorter.sqlite", dry_run: bool = True):
    """Add missing .metadata.json files to the scanned_files table"""
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Find all model directories that have associated files
    cursor.execute('''
        SELECT DISTINCT 
            SUBSTR(source_path, 1, LENGTH(source_path) - LENGTH(REPLACE(source_path, '/', '')) + 1) as dir_path,
            source_path
        FROM model_files 
        WHERE source_path IS NOT NULL
    ''')
    
    model_dirs = {}
    for row in cursor.fetchall():
        # Extract directory path
        model_path = row[1]
        model_dir = os.path.dirname(model_path)
        if model_dir not in model_dirs:
            model_dirs[model_dir] = []
        model_dirs[model_dir].append(model_path)
    
    print(f"Found {len(model_dirs)} unique model directories")
    
    added_files = []
    skipped_files = []
    
    for model_dir, model_files in model_dirs.items():
        # Check if directory exists
        if not os.path.exists(model_dir):
            # Try with /mnt/ instead of /mnt/user/
            alt_dir = model_dir.replace('/mnt/user/', '/mnt/', 1)
            if os.path.exists(alt_dir):
                model_dir = alt_dir
            else:
                continue
        
        # Look for .metadata.json files in the directory
        try:
            for item in os.listdir(model_dir):
                if item.endswith('.metadata.json'):
                    file_path = os.path.join(model_dir, item)
                    
                    # Check if already in database
                    cursor.execute('SELECT id FROM scanned_files WHERE file_path = ?', (file_path,))
                    if cursor.fetchone():
                        skipped_files.append(file_path)
                        continue
                    
                    # Get file stats
                    stat = os.stat(file_path)
                    file_size = stat.st_size
                    last_modified = int(stat.st_mtime)
                    
                    print(f"{'[DRY RUN] ' if dry_run else ''}Adding: {file_path}")
                    
                    if not dry_run:
                        cursor.execute('''
                            INSERT INTO scanned_files 
                            (file_path, file_size, last_modified, file_type, metadata_extracted, scan_timestamp)
                            VALUES (?, ?, ?, 'metadata', 0, ?)
                        ''', (file_path, file_size, last_modified, int(time.time())))
                    
                    added_files.append(file_path)
                    
        except Exception as e:
            print(f"Error processing directory {model_dir}: {e}")
    
    if not dry_run:
        conn.commit()
    
    conn.close()
    
    print(f"\nResults:")
    print(f"  Files added: {len(added_files)}")
    print(f"  Files skipped (already in DB): {len(skipped_files)}")
    
    if dry_run:
        print(f"\nThis was a dry run. Use --execute to actually add files to database.")
    
    return added_files, skipped_files

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Add missing .metadata.json files to scanned_files database")
    parser.add_argument("--database", default="model_sorter.sqlite", help="Database file path")
    parser.add_argument("--execute", action="store_true", help="Actually add files (default is dry run)")
    
    args = parser.parse_args()
    
    dry_run = not args.execute
    add_missing_metadata_files(args.database, dry_run)

if __name__ == "__main__":
    main()