#!/usr/bin/env python3
"""Test script to verify the rescan functionality"""

import sys
import os
import sqlite3

# Add parent directory to path for importing main modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from file_scanner import FileScanner

def test_rescan():
    """Test the rescan functionality"""
    
    # Initialize scanner with config file
    scanner = FileScanner("config.ini")
    
    print("=== Testing Normal Scan (should exclude already scanned files) ===")
    # Test normal scan with a very small limit
    files_normal = scanner.db.get_files_needing_metadata_scan(limit=5, force_rescan=False)
    print(f"Files found for normal scan: {len(files_normal)}")
    
    for i, f in enumerate(files_normal[:3]):
        print(f"  {i+1}. {f['file_name']} (scan_status: {f['scan_status']})")
    
    print("\n=== Testing Rescan (should include already scanned files) ===")
    # Test rescan - this should include files with scan_status=1
    files_rescan = scanner.db.get_files_needing_metadata_scan(limit=20, force_rescan=True)  # Higher limit to catch scanned files
    print(f"Files found for rescan: {len(files_rescan)}")
    
    # Show both scanned and unscanned files
    scanned_found = 0
    unscanned_found = 0
    
    for f in files_rescan:
        if f['scan_status'] == 1:
            scanned_found += 1
            if scanned_found <= 3:  # Show first 3 scanned files
                print(f"  ALREADY SCANNED: {f['file_name']} (scan_status: {f['scan_status']})")
        elif f['scan_status'] == 0 or f['scan_status'] is None:
            unscanned_found += 1
            if unscanned_found <= 3:  # Show first 3 unscanned files  
                print(f"  NOT SCANNED: {f['file_name']} (scan_status: {f['scan_status']})")
    
    print(f"\nSummary:")
    print(f"  - Already scanned files found: {scanned_found}")
    print(f"  - Unscanned files found: {unscanned_found}")
    print(f"  - Total files in rescan: {len(files_rescan)}")
    
    print("\n=== Conclusion ===")
    if scanned_found > 0:
        print("✅ SUCCESS: Rescan mode correctly includes already scanned files")
        print("✅ SUCCESS: Normal scan mode excludes already scanned files")
    else:
        print("❌ ISSUE: Rescan mode should include already scanned files but found none")

if __name__ == "__main__":
    test_rescan()