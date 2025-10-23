#!/usr/bin/env python3
"""
Diagnostic script to find and analyze missing files in the database
"""
import sys
import os
sys.path.append('.')
from file_scanner import FileScanner

def main():
    scanner = FileScanner('.')
    print("🔍 Analyzing database for missing files...")
    
    result = scanner.find_missing_files_and_suggest_cleanup(100)
    
    print(f"\n📊 Analysis Results:")
    print(f"   Total files checked: {result['total_checked']}")
    print(f"   Missing files: {len(result['missing_files'])}")
    print(f"   Possibly moved files: {len(result['moved_files'])}")
    
    if result['missing_files']:
        print(f"\n❌ Missing Files (first 10):")
        for i, missing in enumerate(result['missing_files'][:10]):
            print(f"   {i+1}. {missing['recorded_path']}")
            print(f"      Parent exists: {missing['parent_exists']}")
    
    if result['moved_files']:
        print(f"\n🔄 Possibly Moved Files (first 10):")
        for i, moved in enumerate(result['moved_files'][:10]):
            print(f"   {i+1}. {moved['file_name']}")
            print(f"      Was: {moved['recorded_path']}")
            print(f"      Possible matches: {moved['possible_matches']}")

if __name__ == "__main__":
    main()
