#!/usr/bin/env python3
"""
Script to fix database paths by changing old path prefix to new path prefix
"""
import sqlite3
import os
import sys

def fix_database_paths(db_path: str, old_prefix: str = "/old/path/", new_prefix: str = "/new/path/"):
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
        
        # Count records that need updating
        cursor.execute("SELECT COUNT(*) FROM scanned_files WHERE file_path LIKE ?", (f"{old_prefix}%",))
        scanned_files_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM model_files WHERE source_path LIKE ? OR target_path LIKE ?", 
                      (f"{old_prefix}%", f"{old_prefix}%"))
        model_files_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM associated_files WHERE source_path LIKE ? OR target_path LIKE ?", 
                      (f"{old_prefix}%", f"{old_prefix}%"))
        associated_files_count = cursor.fetchone()[0]
        
        print(f"\nRecords to update:")
        print(f"  - scanned_files: {scanned_files_count:,}")
        print(f"  - model_files: {model_files_count:,}")
        print(f"  - associated_files: {associated_files_count:,}")
        
        if scanned_files_count + model_files_count + associated_files_count == 0:
            print("No records need updating!")
            conn.close()
            return True
        
        # Confirm before proceeding
        response = input(f"\nProceed with updating {scanned_files_count + model_files_count + associated_files_count:,} records? (yes/no): ")
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
                SET file_path = REPLACE(file_path, ?, ?) 
                WHERE file_path LIKE ?
            """, (old_prefix, new_prefix, f"{old_prefix}%"))
            print(f"✅ Updated {cursor.rowcount:,} scanned_files records")
        
        # Fix model_files table
        if model_files_count > 0:
            print(f"Updating {model_files_count:,} model_files records...")
            cursor.execute("""
                UPDATE model_files 
                SET source_path = REPLACE(source_path, ?, ?),
                    target_path = REPLACE(target_path, ?, ?)
                WHERE source_path LIKE ? OR target_path LIKE ?
            """, (old_prefix, new_prefix, old_prefix, new_prefix, f"{old_prefix}%", f"{old_prefix}%"))
            print(f"✅ Updated {cursor.rowcount:,} model_files records")
        
        # Fix associated_files table
        if associated_files_count > 0:
            print(f"Updating {associated_files_count:,} associated_files records...")
            cursor.execute("""
                UPDATE associated_files 
                SET source_path = REPLACE(source_path, ?, ?),
                    target_path = REPLACE(target_path, ?, ?)
                WHERE source_path LIKE ? OR target_path LIKE ?
            """, (old_prefix, new_prefix, old_prefix, new_prefix, f"{old_prefix}%", f"{old_prefix}%"))
            print(f"✅ Updated {cursor.rowcount:,} associated_files records")
        
        # Commit changes
        conn.commit()
        print("\n✅ All database updates committed successfully!")
        
        # Verify updates
        print("\nVerifying updates...")
        cursor.execute("SELECT COUNT(*) FROM scanned_files WHERE file_path LIKE ?", (f"{old_prefix}%",))
        remaining_scanned = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM model_files WHERE source_path LIKE ? OR target_path LIKE ?", 
                      (f"{old_prefix}%", f"{old_prefix}%"))
        remaining_model = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM associated_files WHERE source_path LIKE ? OR target_path LIKE ?", 
                      (f"{old_prefix}%", f"{old_prefix}%"))
        remaining_associated = cursor.fetchone()[0]
        
        total_remaining = remaining_scanned + remaining_model + remaining_associated
        
        if total_remaining == 0:
            print("✅ Verification complete - all paths successfully updated!")
        else:
            print(f"⚠️  Warning: {total_remaining} records still have old paths")
            print(f"   - scanned_files: {remaining_scanned}")
            print(f"   - model_files: {remaining_model}")
            print(f"   - associated_files: {remaining_associated}")
        
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
    
    print("Database Path Fix Script")
    print("=" * 50)
    
    success = fix_database_paths(db_path)
    sys.exit(0 if success else 1)