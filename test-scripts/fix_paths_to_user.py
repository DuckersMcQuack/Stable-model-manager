#!/usr/bin/env python3
"""
Script to fix database paths by changing old path prefix to new path prefix
This corrects the database to match the current server mount structure.
"""
import sqlite3
import os
import sys

def fix_database_paths_to_user(db_path: str, old_prefix: str = "/old/path/", new_prefix: str = "/new/path/"):
    """Fix all database paths from old prefix to new prefix"""
    
    if not os.path.exists(db_path):
        print(f"ERROR: Database not found at {db_path}")
        return False
    
    try:
        # Connect to database with write permissions
        conn = sqlite3.connect(db_path, timeout=30.0)
        cursor = conn.cursor()
        
        print(f"Connected to database: {db_path}")
        print(f"Fixing paths: {old_prefix} -> {new_prefix}")
        
        # Special handling to avoid double conversion
        # We want to change /old/path/Models to /new/path/Models
        # But NOT change /new/path/Models to /new/path/new/path/Models
        
        # Count records that need updating (exclude paths that already have the new prefix)
        # Only target specific model archive paths
        cursor.execute("""
            SELECT COUNT(*) FROM scanned_files 
            WHERE file_path LIKE '/old/path/Models/%' 
            AND file_path NOT LIKE '/new/path/%'
        """)
        scanned_files_count = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT COUNT(*) FROM model_files 
            WHERE (source_path LIKE '/old/path/Models/%' AND source_path NOT LIKE '/new/path/%')
            OR (target_path LIKE '/old/path/Models/%' AND target_path NOT LIKE '/new/path/%')
        """)
        model_files_count = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT COUNT(*) FROM associated_files 
            WHERE (source_path LIKE '/mnt/AI-Dventure/Civitai-Archive/%' AND source_path NOT LIKE '/mnt/user/%')
            OR (target_path LIKE '/mnt/AI-Dventure/Civitai-Archive/%' AND target_path NOT LIKE '/mnt/user/%')
        """)
        associated_files_count = cursor.fetchone()[0]
        
        print(f"\nRecords to update:")
        print(f"  - scanned_files: {scanned_files_count:,}")
        print(f"  - model_files: {model_files_count:,}")
        print(f"  - associated_files: {associated_files_count:,}")
        
        if scanned_files_count + model_files_count + associated_files_count == 0:
            print("No records need updating!")
            conn.close()
            return True
        
        # Show some examples of what will be changed
        print("\nExample paths that will be updated:")
        cursor.execute("""
            SELECT file_path FROM scanned_files 
            WHERE file_path LIKE '/mnt/AI-Dventure/Civitai-Archive/%' 
            AND file_path NOT LIKE '/mnt/user/%'
            LIMIT 3
        """)
        examples = cursor.fetchall()
        for example in examples:
            old_path = example[0]
            new_path = old_path.replace('/mnt/', '/mnt/user/', 1)
            print(f"  {old_path}")
            print(f"  -> {new_path}")
            print()
        
        # Confirm before proceeding
        total_records = scanned_files_count + model_files_count + associated_files_count
        response = input(f"Proceed with updating {total_records:,} records? (yes/no): ")
        if response.lower() != 'yes':
            print("Aborted by user")
            conn.close()
            return False
        
        # Begin transaction
        print("\nStarting database updates...")
        
        # Fix scanned_files table
        if scanned_files_count > 0:
            print(f"Updating {scanned_files_count:,} scanned_files records...")
            cursor.execute("""
                UPDATE scanned_files 
                SET file_path = REPLACE(file_path, '/mnt/', '/mnt/user/') 
                WHERE file_path LIKE '/mnt/AI-Dventure/Civitai-Archive/%' 
                AND file_path NOT LIKE '/mnt/user/%'
            """)
            print(f"✅ Updated {cursor.rowcount:,} scanned_files records")
        
        # Fix model_files table
        if model_files_count > 0:
            print(f"Updating model_files records...")
            
            # Update source_path
            cursor.execute("""
                UPDATE model_files 
                SET source_path = REPLACE(source_path, '/mnt/', '/mnt/user/')
                WHERE source_path LIKE '/mnt/AI-Dventure/Civitai-Archive/%' 
                AND source_path NOT LIKE '/mnt/user/%'
            """)
            source_updated = cursor.rowcount
            
            # Update target_path
            cursor.execute("""
                UPDATE model_files 
                SET target_path = REPLACE(target_path, '/mnt/', '/mnt/user/')
                WHERE target_path LIKE '/mnt/AI-Dventure/Civitai-Archive/%' 
                AND target_path NOT LIKE '/mnt/user/%'
            """)
            target_updated = cursor.rowcount
            
            print(f"✅ Updated {source_updated:,} source paths and {target_updated:,} target paths in model_files")
        
        # Fix associated_files table
        if associated_files_count > 0:
            print(f"Updating associated_files records...")
            
            # Update source_path
            cursor.execute("""
                UPDATE associated_files 
                SET source_path = REPLACE(source_path, '/mnt/', '/mnt/user/')
                WHERE source_path LIKE '/mnt/AI-Dventure/Civitai-Archive/%' 
                AND source_path NOT LIKE '/mnt/user/%'
            """)
            source_updated = cursor.rowcount
            
            # Update target_path
            cursor.execute("""
                UPDATE associated_files 
                SET target_path = REPLACE(target_path, '/mnt/', '/mnt/user/')
                WHERE target_path LIKE '/mnt/AI-Dventure/Civitai-Archive/%' 
                AND target_path NOT LIKE '/mnt/user/%'
            """)
            target_updated = cursor.rowcount
            
            print(f"✅ Updated {source_updated:,} source paths and {target_updated:,} target paths in associated_files")
        
        # Commit changes
        conn.commit()
        print("\n✅ All database updates committed successfully!")
        
        # Verify updates
        print("\nVerifying updates...")
        cursor.execute("""
            SELECT COUNT(*) FROM scanned_files 
            WHERE file_path LIKE '/mnt/AI-Dventure/Civitai-Archive/%' 
            AND file_path NOT LIKE '/mnt/user/%'
        """)
        remaining_scanned = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT COUNT(*) FROM model_files 
            WHERE (source_path LIKE '/mnt/AI-Dventure/Civitai-Archive/%' AND source_path NOT LIKE '/mnt/user/%')
            OR (target_path LIKE '/mnt/AI-Dventure/Civitai-Archive/%' AND target_path NOT LIKE '/mnt/user/%')
        """)
        remaining_model = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT COUNT(*) FROM associated_files 
            WHERE (source_path LIKE '/mnt/AI-Dventure/Civitai-Archive/%' AND source_path NOT LIKE '/mnt/user/%')
            OR (target_path LIKE '/mnt/AI-Dventure/Civitai-Archive/%' AND target_path NOT LIKE '/mnt/user/%')
        """)
        remaining_associated = cursor.fetchone()[0]
        
        total_remaining = remaining_scanned + remaining_model + remaining_associated
        
        if total_remaining == 0:
            print("✅ Verification complete - all paths successfully updated!")
        else:
            print(f"⚠️  Warning: {total_remaining} records still have old paths")
            print(f"   - scanned_files: {remaining_scanned}")
            print(f"   - model_files: {remaining_model}")
            print(f"   - associated_files: {remaining_associated}")
        
        # Show final statistics
        cursor.execute("SELECT COUNT(*) FROM scanned_files WHERE file_path LIKE '/mnt/user/%'")
        final_scanned = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT COUNT(*) FROM model_files 
            WHERE source_path LIKE '/mnt/user/%' OR target_path LIKE '/mnt/user/%'
        """)
        final_model = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT COUNT(*) FROM associated_files 
            WHERE source_path LIKE '/mnt/user/%' OR target_path LIKE '/mnt/user/%'
        """)
        final_associated = cursor.fetchone()[0]
        
        print(f"\nFinal statistics with /mnt/user/ paths:")
        print(f"  - scanned_files: {final_scanned:,}")
        print(f"  - model_files: {final_model:,}")
        print(f"  - associated_files: {final_associated:,}")
        
        conn.close()
        return True
        
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

if __name__ == "__main__":
    db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_sorter.sqlite")
    
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    
    print("Database Path Fix Script - Update path prefixes")
    print("=" * 70)
    print("This script changes /old/path/Models/...")
    print("                 -> /new/path/Models/...")
    print("=" * 70)
    
    success = fix_database_paths_to_user(db_path)
    sys.exit(0 if success else 1)