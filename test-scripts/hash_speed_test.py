#!/usr/bin/env python3
"""
Hash Speed Benchmark Script

Tests different hashing methods on large files to identify performance bottlenecks.
Specifically designed to test the double-read issue in SafeTensors AutoV3 calculation.
"""

import hashlib
import time
import os
from pathlib import Path
from typing import Optional, Tuple

# Try to import blake3 - install with: pip install blake3
try:
    import blake3  # type: ignore
    BLAKE3_AVAILABLE = True
except ImportError:
    BLAKE3_AVAILABLE = False
    print("⚠️  BLAKE3 not available. Install with: pip install blake3")

def format_speed(file_size_bytes: int, time_seconds: float) -> str:
    """Format speed in MB/s"""
    if time_seconds == 0:
        return "∞ MB/s"
    mb_size = file_size_bytes / (1024 * 1024)
    speed = mb_size / time_seconds
    return f"{speed:.1f} MB/s"

def format_time(seconds: float) -> str:
    """Format time in appropriate units"""
    if seconds < 1:
        return f"{seconds*1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"

def test_sha256_only(file_path: str) -> Tuple[str, float]:
    """Test SHA256 hashing only (current method 1)"""
    print("🔹 Testing SHA256 only...")
    
    start_time = time.time()
    hash_sha256 = hashlib.sha256()
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hash_sha256.update(chunk)
    
    end_time = time.time()
    return hash_sha256.hexdigest().upper(), end_time - start_time

def test_blake3_only(file_path: str) -> Tuple[Optional[str], float]:
    """Test BLAKE3 hashing only (fast alternative for non-SafeTensors)"""
    print("🔹 Testing BLAKE3 only...")
    
    if not BLAKE3_AVAILABLE:
        return None, 0.0
    
    start_time = time.time()
    
    try:
        hasher = blake3.blake3()  # type: ignore
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        
        end_time = time.time()
        return hasher.hexdigest().upper(), end_time - start_time
    except Exception:
        return None, time.time() - start_time

def test_autov3_only(file_path: str) -> Tuple[Optional[str], float]:
    """Test AutoV3 hashing only (SafeTensors-specific method)"""
    print("🔹 Testing AutoV3 only (SafeTensors-specific)...")
    
    start_time = time.time()
    
    try:
        with open(file_path, "rb") as f:
            header8 = f.read(8)
            if len(header8) < 8:
                return None, time.time() - start_time
            
            header_size = int.from_bytes(header8, "little")
            offset = header_size + 8
            
            # Check file size
            f.seek(0, 2)
            filesize = f.tell()
            if offset >= filesize:
                return None, time.time() - start_time
            
            # Compute sha256 starting at offset
            f.seek(offset)
            h = hashlib.sha256()
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
            
            end_time = time.time()
            return h.hexdigest().upper(), end_time - start_time
    except Exception:
        return None, time.time() - start_time

def test_double_read_current(file_path: str) -> Tuple[str, Optional[str], float]:
    """Test current double-read approach (SHA256 + AutoV3 separately)"""
    print("🔹 Testing current double-read method...")
    
    start_time = time.time()
    
    # First pass: SHA256
    hash_sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hash_sha256.update(chunk)
    sha256_hex = hash_sha256.hexdigest().upper()
    
    # Second pass: AutoV3 (if SafeTensors-like)
    autov3_hash = None
    try:
        with open(file_path, "rb") as f:
            header8 = f.read(8)
            if len(header8) == 8:
                header_size = int.from_bytes(header8, "little")
                offset = header_size + 8
                f.seek(0, 2)
                filesize = f.tell()
                if offset < filesize:
                    f.seek(offset)
                    h = hashlib.sha256()
                    for chunk in iter(lambda: f.read(8192), b""):
                        h.update(chunk)
                    autov3_hash = h.hexdigest().upper()
    except Exception:
        pass
    
    end_time = time.time()
    return sha256_hex, autov3_hash, end_time - start_time

def test_single_read_optimized(file_path: str) -> Tuple[str, Optional[str], float]:
    """Test optimized single-read approach"""
    print("🔹 Testing optimized single-read method...")
    
    start_time = time.time()
    
    hash_sha256 = hashlib.sha256()
    autov3_hash = None
    
    try:
        with open(file_path, 'rb') as f:
            # Try to read header for SafeTensors detection
            header8 = f.read(8)
            offset = 0
            autov3_hasher = None
            
            if len(header8) == 8:
                try:
                    header_size = int.from_bytes(header8, "little")
                    offset = header_size + 8
                    f.seek(0, 2)
                    filesize = f.tell()
                    if offset < filesize:
                        autov3_hasher = hashlib.sha256()
                except:
                    pass
            
            # Reset to beginning for full file read
            f.seek(0)
            bytes_read = 0
            
            # Single pass through file
            for chunk in iter(lambda: f.read(65536), b""):
                # Always update SHA256 with full file
                hash_sha256.update(chunk)
                
                # For AutoV3, only update with bytes after offset
                if autov3_hasher and bytes_read >= offset:
                    # Entire chunk is after offset
                    autov3_hasher.update(chunk)
                elif autov3_hasher and bytes_read + len(chunk) > offset:
                    # Chunk spans the offset - use partial chunk
                    chunk_start = offset - bytes_read
                    autov3_hasher.update(chunk[chunk_start:])
                
                bytes_read += len(chunk)
            
            sha256_hex = hash_sha256.hexdigest().upper()
            if autov3_hasher:
                autov3_hash = autov3_hasher.hexdigest().upper()
            
    except Exception as e:
        print(f"Error in optimized method: {e}")
        sha256_hex = hash_sha256.hexdigest().upper()
    
    end_time = time.time()
    return sha256_hex, autov3_hash, end_time - start_time

def test_different_chunk_sizes(file_path: str) -> None:
    """Test different chunk sizes for reading"""
    print("\n📊 Testing different chunk sizes for SHA256:")
    
    chunk_sizes = [8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]  # 8KB to 1MB
    file_size = os.path.getsize(file_path)
    
    for chunk_size in chunk_sizes:
        start_time = time.time()
        hash_sha256 = hashlib.sha256()
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                hash_sha256.update(chunk)
        
        end_time = time.time()
        duration = end_time - start_time
        speed = format_speed(file_size, duration)
        
        print(f"  {chunk_size//1024:4d}KB chunks: {format_time(duration):>8} ({speed:>10})")

def test_blake3_chunk_sizes(file_path: str) -> None:
    """Test different chunk sizes for BLAKE3"""
    if not BLAKE3_AVAILABLE:
        return
        
    print("\n📊 Testing different chunk sizes for BLAKE3:")
    
    chunk_sizes = [8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]  # 8KB to 1MB
    file_size = os.path.getsize(file_path)
    
    for chunk_size in chunk_sizes:
        start_time = time.time()
        hasher = blake3.blake3()  # type: ignore
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                hasher.update(chunk)
        
        end_time = time.time()
        duration = end_time - start_time
        speed = format_speed(file_size, duration)
        
        print(f"  {chunk_size//1024:4d}KB chunks: {format_time(duration):>8} ({speed:>10})")

def main():
    # Look for .safetensors files in sample_models directory first, then current directory
    sample_models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'sample_models')
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    safetensors_files = []
    test_dir = None
    
    # Check sample_models directory first
    if os.path.exists(sample_models_dir):
        sample_files = [f for f in os.listdir(sample_models_dir) if f.endswith('.safetensors')]
        if sample_files:
            safetensors_files = sample_files
            test_dir = sample_models_dir
            print(f"📁 Using files from sample_models/ directory")
    
    # Fallback to current directory 
    if not safetensors_files:
        current_files = [f for f in os.listdir(current_dir) if f.endswith('.safetensors')]
        if current_files:
            safetensors_files = current_files
            test_dir = current_dir
            print(f"� Using files from test-scripts/ directory")
    
    if not safetensors_files:
        print(f"❌ No .safetensors files found in sample_models/ or test-scripts/")
        print("💡 Add .safetensors files to sample_models/ directory for benchmarking")
        print("   Alternatively, copy files directly to test-scripts/ directory")
        return
    
    # Use the first .safetensors file found
    test_file = os.path.join(test_dir or current_dir, safetensors_files[0])
    print(f"🔍 Found {len(safetensors_files)} .safetensors file(s)")
    print(f"📂 Using: {safetensors_files[0]}")
    
    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        return
    
    file_size = os.path.getsize(test_file)
    print(f"🎯 Hash Speed Benchmark")
    print(f"📁 Test file: {test_file}")
    print(f"📏 File size: {file_size:,} bytes ({file_size / (1024**3):.2f} GB)")
    print()
    print("📋 Hash Usage Strategy:")
    print("   • SHA256: Universal hash for all files (compatibility)")
    print("   • BLAKE3: Fast alternative for non-SafeTensors files")  
    print("   • AutoV3: SafeTensors-specific (tensor data only)")
    print("=" * 80)
    
    results = {}
    
    # Test 1: SHA256 only
    sha256_hash, sha256_time = test_sha256_only(test_file)
    results['SHA256 only'] = (sha256_time, format_speed(file_size, sha256_time))
    print(f"   ✓ SHA256: {sha256_hash[:16]}...")
    print(f"   ⏱️  Time: {format_time(sha256_time)} ({format_speed(file_size, sha256_time)})")
    print()
    
    # Test 2: BLAKE3 only
    blake3_hash, blake3_time = test_blake3_only(test_file)
    if BLAKE3_AVAILABLE:
        results['BLAKE3 only'] = (blake3_time, format_speed(file_size, blake3_time))
        if blake3_hash:
            print(f"   ✓ BLAKE3: {blake3_hash[:16]}...")
        print(f"   ⏱️  Time: {format_time(blake3_time)} ({format_speed(file_size, blake3_time)})")
    else:
        print(f"   ⚠️  BLAKE3: Not available (install with: pip install blake3)")
    print()

    # Test 3: AutoV3 only (SafeTensors-specific)
    autov3_hash, autov3_time = test_autov3_only(test_file)
    results['AutoV3 only (SafeTensors)'] = (autov3_time, format_speed(file_size, autov3_time))
    if autov3_hash:
        print(f"   ✓ AutoV3: {autov3_hash[:16]}...")
    else:
        print(f"   ⚠️  AutoV3: Not SafeTensors format or invalid header")
    print(f"   ⏱️  Time: {format_time(autov3_time)} ({format_speed(file_size, autov3_time)})")
    print()
    
    # Test 4: Current double-read method
    sha256_double, autov3_double, double_time = test_double_read_current(test_file)
    results['Double-read (current)'] = (double_time, format_speed(file_size, double_time))
    print(f"   ✓ SHA256: {sha256_double[:16]}...")
    if autov3_double:
        print(f"   ✓ AutoV3: {autov3_double[:16]}...")
    else:
        print(f"   ⚠️  AutoV3: Not SafeTensors format")
    print(f"   ⏱️  Time: {format_time(double_time)} ({format_speed(file_size, double_time)})")
    print()
    
    # Test 5: Optimized single-read method
    sha256_single, autov3_single, single_time = test_single_read_optimized(test_file)
    results['Single-read (optimized)'] = (single_time, format_speed(file_size, single_time))
    print(f"   ✓ SHA256: {sha256_single[:16]}...")
    if autov3_single:
        print(f"   ✓ AutoV3: {autov3_single[:16]}...")
    else:
        print(f"   ⚠️  AutoV3: Not SafeTensors format")
    print(f"   ⏱️  Time: {format_time(single_time)} ({format_speed(file_size, single_time)})")
    print()
    
    # Verify hashes match
    print("🔍 Hash Verification:")
    sha256_match = sha256_hash == sha256_double == sha256_single
    print(f"   SHA256 consistency: {'✅' if sha256_match else '❌'}")
    
    if BLAKE3_AVAILABLE and blake3_hash:
        print(f"   BLAKE3 calculated: ✅")
    
    if autov3_hash and autov3_double and autov3_single:
        autov3_match = autov3_hash == autov3_double == autov3_single
        print(f"   AutoV3 consistency: {'✅' if autov3_match else '❌'}")
    elif autov3_hash:
        print(f"   AutoV3 (SafeTensors): ✅")
    
    # Performance summary
    print("\n📈 Performance Summary:")
    print("-" * 60)
    for method, (time_sec, speed) in results.items():
        print(f"{method:25} {format_time(time_sec):>10} {speed:>12}")
    
    # Calculate improvements
    if double_time > 0:
        improvement = ((double_time - single_time) / double_time) * 100
        print(f"\n🚀 Single-read improvement: {improvement:+.1f}%")
        
        if improvement > 0:
            print(f"   💡 Optimized method is {improvement:.1f}% faster!")
        else:
            print(f"   ⚠️  Current method is {abs(improvement):.1f}% faster")
    
    # Test chunk sizes
    test_different_chunk_sizes(test_file)
    test_blake3_chunk_sizes(test_file)

if __name__ == "__main__":
    main()