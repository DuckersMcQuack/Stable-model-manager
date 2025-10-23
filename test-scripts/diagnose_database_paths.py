#!/usr/bin/env python3
"""
Diagnostic script to examine actual paths in the database
"""
import sqlite3
import os
import sys

def diagnose_database_paths(db_path: str):
    """Examine what paths actually exist in the database"""
    
    if not os.path.exists(db_path):
        print(f"ERROR: Database not found at {db_path}")
        return False
    
    try:
        conn = sqlite3.connect(db_path, timeout=30.0)
        cursor = conn.cursor()
        
        print(f"Connected to database: {db_path}")
        print("=" * 70)
        
        # Sample paths from each table
        print("\n1. SCANNED_FILES - Sample paths:")
        cursor.execute("SELECT file_path FROM scanned_files LIMIT 10")
        scanned_samples = cursor.fetchall()
        for i, (path,) in enumerate(scanned_samples, 1):
            print(f"   {i:2d}. {path}")
        
        print("\n2. MODEL_FILES - Sample source paths:")
        cursor.execute("SELECT source_path FROM model_files WHERE source_path IS NOT NULL LIMIT 10")
        model_samples = cursor.fetchall()
        for i, (path,) in enumerate(model_samples, 1):
            print(f"   {i:2d}. {path}")
        
        print("\n3. ASSOCIATED_FILES - Sample source paths:")
        cursor.execute("SELECT source_path FROM associated_files WHERE source_path IS NOT NULL LIMIT 10")
        assoc_samples = cursor.fetchall()
        for i, (path,) in enumerate(assoc_samples, 1):
            print(f"   {i:2d}. {path}")
        
        # Count different path patterns
        print("\n" + "=" * 70)
        print("PATH PATTERN ANALYSIS:")
        
        # Scanned files patterns
        cursor.execute("SELECT COUNT(*) FROM scanned_files WHERE file_path LIKE '/mnt/user/%'")
        user_paths = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM scanned_files WHERE file_path LIKE '/mnt/%' AND file_path NOT LIKE '/mnt/user/%'")
        direct_mnt_paths = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM scanned_files WHERE file_path LIKE '/mnt/user/%'")
        ai_dventure_paths = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM scanned_files WHERE file_path LIKE '/mnt/user/AI-Dventure/%'")
        user_ai_dventure_paths = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM scanned_files WHERE file_path LIKE '/mnt/user/Civitai-Archive/%'")
        direct_civitai_paths = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM scanned_files WHERE file_path LIKE '/mnt/user/AI-Dventure/Civitai-Archive/%'")
        user_civitai_paths = cursor.fetchone()[0]
        
        print(f"\nSCANNED_FILES patterns:")
        print(f"  /mnt/user/... paths: {user_paths:,}")
        print(f"  /mnt/... (no user) paths: {direct_mnt_paths:,}")
        print(f"  Old path patterns: {ai_dventure_paths:,}")
        print(f"  /mnt/user/Archive/... paths: {user_ai_dventure_paths:,}")
        print(f"  Direct archive paths: {direct_civitai_paths:,}")
        print(f"  /mnt/user/Archive/... paths: {user_civitai_paths:,}")
        
        # Model files patterns
        cursor.execute("SELECT COUNT(*) FROM model_files WHERE source_path LIKE '/mnt/user/%'")
        model_user_paths = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM model_files WHERE source_path LIKE '/mnt/%' AND source_path NOT LIKE '/mnt/user/%'")
        model_direct_paths = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM model_files WHERE source_path LIKE '/mnt/user/Archive/%'")
        model_direct_civitai = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM model_files WHERE source_path LIKE '/mnt/user/AI-Dventure/Civitai-Archive/%'")
        model_user_civitai = cursor.fetchone()[0]
        
        print(f"\nMODEL_FILES patterns:")
        print(f"  /mnt/user/... paths: {model_user_paths:,}")
        print(f"  /mnt/... (no user) paths: {model_direct_paths:,}")
        print(f"  /mnt/AI-Dventure/Civitai-Archive/... paths: {model_direct_civitai:,}")
        print(f"  /mnt/user/AI-Dventure/Civitai-Archive/... paths: {model_user_civitai:,}")
        
        # Associated files patterns
        cursor.execute("SELECT COUNT(*) FROM associated_files WHERE source_path LIKE '/mnt/user/%'")
        assoc_user_paths = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM associated_files WHERE source_path LIKE '/mnt/%' AND source_path NOT LIKE '/mnt/user/%'")
        assoc_direct_paths = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM associated_files WHERE source_path LIKE '/mnt/user/Archive/%'")
        assoc_direct_civitai = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM associated_files WHERE source_path LIKE '/mnt/user/AI-Dventure/Civitai-Archive/%'")
        assoc_user_civitai = cursor.fetchone()[0]
        
        print(f"\nASSOCIATED_FILES patterns:")
        print(f"  /mnt/user/... paths: {assoc_user_paths:,}")
        print(f"  /mnt/... (no user) paths: {assoc_direct_paths:,}")
        print(f"  /mnt/AI-Dventure/Civitai-Archive/... paths: {assoc_direct_civitai:,}")
        print(f"  /mnt/user/AI-Dventure/Civitai-Archive/... paths: {assoc_user_civitai:,}")
        
        # Look for the specific error case
        print("\n" + "=" * 70)
        print("SPECIFIC ERROR CASE ANALYSIS:")
        
        cursor.execute("""
            SELECT source_path FROM associated_files 
            WHERE source_path LIKE '%kotatsu_v1_pony.txt' 
            OR source_path LIKE '%chrisevans%metadata%'
            LIMIT 5
        """)
        error_cases = cursor.fetchall()
        
        if error_cases:
            print("Found paths matching recent error cases:")
            for path, in error_cases:
                print(f"  {path}")
        else:
            print("No paths found matching recent error cases")
        
        # Check for any other mount patterns
        print("\n" + "=" * 70)
        print("ALL MOUNT PATTERNS:")
        
        cursor.execute("""
            SELECT DISTINCT 
                CASE 
                    WHEN file_path LIKE '/mnt/user/%' THEN '/mnt/user/...'
                    WHEN file_path LIKE '/mnt/%' THEN '/mnt/...'
                    WHEN file_path LIKE '/home/%' THEN '/home/...'
                    ELSE 'Other: ' || SUBSTR(file_path, 1, 20) || '...'
                END as pattern,
                COUNT(*) as count
            FROM scanned_files 
            GROUP BY pattern
            ORDER BY count DESC
        """)
        patterns = cursor.fetchall()
        
        print("All path patterns in scanned_files:")
        for pattern, count in patterns:
            print(f"  {pattern}: {count:,} files")
        
        conn.close()
        return True
        
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

if __name__ == "__main__":
    db_path = "/path/to/Stable-model-manager/model_sorter.sqlite"
    
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    
    print("Database Path Diagnostic Script")
    print("=" * 70)
    
    success = diagnose_database_paths(db_path)
    sys.exit(0 if success else 1)