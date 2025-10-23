#!/usr/bin/env python3
"""
Debug script to isolate CIFS performance issues during LoRA scanning
"""

import os
import time
import hashlib
from pathlib import Path

def test_file_operations(file_path):
    """Test various file operations to identify bottlenecks"""
    
    print(f"Testing file: {file_path}")
    print("=" * 80)
    
    # Test 1: File stat
    start = time.time()
    try:
        stat_info = os.stat(file_path)
        stat_time = time.time() - start
        print(f"✓ os.stat(): {stat_time:.3f}s (size: {stat_info.st_size:,} bytes)")
    except Exception as e:
        print(f"✗ os.stat() failed: {e}")
        return
    
    # Test 2: File open
    start = time.time()
    try:
        with open(file_path, 'rb') as f:
            pass
        open_time = time.time() - start
        print(f"✓ File open: {open_time:.3f}s")
    except Exception as e:
        print(f"✗ File open failed: {e}")
        return
    
    # Test 3: Read first 8 bytes (SafeTensors header)
    start = time.time()
    try:
        with open(file_path, 'rb') as f:
            header = f.read(8)
            header_time = time.time() - start
            print(f"✓ Header read (8 bytes): {header_time:.3f}s")
            
            if len(header) == 8:
                header_size = int.from_bytes(header, "little")
                print(f"  SafeTensors header size: {header_size:,} bytes")
    except Exception as e:
        print(f"✗ Header read failed: {e}")
        return
    
    # Test 4: Read first chunk (65KB)
    start = time.time()
    try:
        with open(file_path, 'rb') as f:
            chunk = f.read(65536)
            chunk_time = time.time() - start
            print(f"✓ First chunk read (65KB): {chunk_time:.3f}s")
    except Exception as e:
        print(f"✗ First chunk read failed: {e}")
        return
    
    # Test 5: Full SHA256 calculation
    start = time.time()
    try:
        hash_sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while chunk := f.read(65536):
                hash_sha256.update(chunk)
        sha256_hex = hash_sha256.hexdigest()
        sha256_time = time.time() - start
        file_size_mb = stat_info.st_size / (1024 * 1024)
        speed_mb_s = file_size_mb / sha256_time if sha256_time > 0 else 0
        print(f"✓ Full SHA256: {sha256_time:.3f}s ({speed_mb_s:.1f} MB/s)")
        print(f"  Hash: {sha256_hex[:16]}...")
    except Exception as e:
        print(f"✗ SHA256 calculation failed: {e}")
        return
    
    # Test 6: SafeTensors AutoV3 calculation
    start = time.time()
    try:
        with open(file_path, 'rb') as f:
            header8 = f.read(8)
            if len(header8) == 8:
                header_size = int.from_bytes(header8, "little")
                offset = header_size + 8
                f.seek(0, 2)  # Seek to end
                seek_time = time.time()
                filesize = f.tell()
                f.seek(offset)  # Seek to data start
                seek2_time = time.time()
                
                h = hashlib.sha256()
                while chunk := f.read(8192):
                    h.update(chunk)
                autov3_hex = h.hexdigest()
                autov3_time = time.time() - start
                
                print(f"✓ AutoV3 hash: {autov3_time:.3f}s")
                print(f"  Seeks: end={seek_time-start:.3f}s, offset={seek2_time-seek_time:.3f}s")
                print(f"  Hash: {autov3_hex[:16]}...")
    except Exception as e:
        print(f"✗ AutoV3 calculation failed: {e}")
    
    print("=" * 80)

if __name__ == "__main__":
    test_file = "/path/to/models/SD 1.5/sample_character_lora_v1.0.safetensors"
    
    if os.path.exists(test_file):
        test_file_operations(test_file)
    else:
        print(f"Test file not found: {test_file}")
        
    # Test a few more files if available  
    lora_dir = "sample_models/"  # Update with your test directory
    if os.path.exists(lora_dir):
        print(f"\nTesting additional files in: {lora_dir}")
        safetensors_files = [f for f in os.listdir(lora_dir) if f.endswith('.safetensors')][:3]
        for filename in safetensors_files:
            filepath = os.path.join(lora_dir, filename)
            print(f"\nQuick test: {filename}")
            start = time.time()
            try:
                stat_info = os.stat(filepath)
                stat_time = time.time() - start
                print(f"  os.stat(): {stat_time:.3f}s ({stat_info.st_size / (1024*1024):.1f} MB)")
            except Exception as e:
                print(f"  os.stat() failed: {e}")