#!/usr/bin/env python3
"""
Test utilities for test scripts - provides common functions for finding and working with test files
"""

import os
import sys
from pathlib import Path
from typing import List, Optional

def get_safetensors_files() -> List[str]:
    """Get list of .safetensors files in the current test directory"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return [f for f in os.listdir(current_dir) if f.endswith('.safetensors')]

def get_first_safetensors() -> Optional[str]:
    """Get the full path to the first .safetensors file found"""
    files = get_safetensors_files()
    if not files:
        return None
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, files[0])

def setup_parent_import():
    """Add parent directory to path so we can import from the main scripts"""
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

def print_available_files():
    """Print all available .safetensors files for testing"""
    files = get_safetensors_files()
    if files:
        print(f"📂 Available .safetensors files ({len(files)}):")
        for i, f in enumerate(files, 1):
            size = os.path.getsize(os.path.join(os.path.dirname(os.path.abspath(__file__)), f))
            print(f"   {i}. {f} ({size:,} bytes)")
    else:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"❌ No .safetensors files found in: {current_dir}")
        print("💡 Copy some .safetensors files to this directory for testing")

if __name__ == "__main__":
    print_available_files()