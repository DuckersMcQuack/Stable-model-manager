# Test Scripts Directory

This directory contains all testing, verification, benchmark, and diagnostic scripts for the Model Organizer project.

## 🚀 Quick Start

The test scripts now **automatically discover and use files** from the sample directories:

1. **Add your test files** to the appropriate sample directories:
   - `../sample_models/` - Add `.safetensors`, `.ckpt`, `.pt` model files
   - `../sample_media/` - Add `.png`, `.jpg`, `.webp` images
   - `../sample_metadata/` - Add `.json`, `.txt`, `.civitai.info` files

2. **Run comprehensive tests**:
   ```bash
   python comprehensive_integration_test.py
   ```

3. **Run specific tests**:
   ```bash
   python test_metadata.py          # Test metadata extraction
   python test_folder_detection.py  # Test base model detection
   python hash_speed_test.py        # Benchmark hash performance
   ```

## 📁 Dynamic Testing System

### Sample Directory Integration
All test scripts now **automatically scan** the sample directories and use actual files for testing:

- **No hardcoded filenames** - Tests discover files dynamically
- **Real file testing** - Uses your actual test files instead of synthetic data
- **Graceful fallbacks** - Works with empty directories, provides helpful guidance
- **Latest code integration** - Uses current versions of all main modules

### Key Test Scripts

- **`comprehensive_integration_test.py`** - 🌟 **MAIN TEST** - Complete pipeline testing
  - Automatically discovers files in sample directories
  - Tests FileScanner, MetadataExtractor, DuplicateDetector integration
  - Provides detailed results and recommendations
  - Perfect for validating your test environment

### Core Functionality Tests
- **`test_metadata.py`** - Metadata extraction from images and models
- **`test_folder_detection.py`** - Base model detection from folder paths
- **`test_batch_commit.py`** - Database batch operations
- **`test_metadata_extraction.py`** - Advanced metadata parsing

### Benchmark Scripts
- **`hash_speed_test.py`** - Hash performance benchmarking
  - Automatically scans for .safetensors files in directory
  - Tests SHA256, BLAKE3, and AutoV3 hash performance
  - Compares single-read vs double-read methods
  - Tests different chunk sizes for optimization

### Verification Scripts
- **`test_autov3.py`** - AutoV3 hash computation verification
- **`test_metadata.py`** - Metadata extraction testing
- **`test_batch_commit.py`** - Database batch commit testing
- **`test_rescan.py`** - File rescanning functionality
- **`test_retry_failed.py`** - Failed operation retry testing
- **`feature_verification.py`** - Complete feature testing suite
- **`final_verification.py`** - Final system verification check

### Diagnostic Scripts
- **`database_inspector.py`** - Database structure and content inspection
- **`diagnose_database_paths.py`** - Database path issue diagnosis
- **`diagnose_missing_files.py`** - Missing file detection and recovery

### Utility Scripts (Maintenance)
- **`add_missing_metadata_files.py`** - Add missing metadata entries to database
- **`fix_database_paths.py`** - Fix database path inconsistencies
- **`fix_paths_to_user.py`** - Migrate paths from old to new prefixes
- **`update_database_schema.py`** - Update database schema versions
- **`view_report.py`** - View processing reports and statistics

### Performance & Debug Scripts
- **`debug_cifs_performance.py`** - Network file system performance analysis
- **`fast_prescan_optimization.py`** - Prescan performance optimization testing
- **`optimized_db_lookup.py`** - Database lookup performance testing

### Demo Scripts
- **`demo_cross_reference.py`** - Showcase cross-referencing capabilities
- **`demo_incremental_complete.py`** - Demonstrate incremental processing
- **`metadata_extractor_cli.py`** - CLI wrapper for metadata extraction

### Documentation
- **`blake3_integration_docs.py`** - BLAKE3 hash integration documentation

### Backup Files
- **`model_sorter_main_backup.py`** - Backup of main script
- **`model_sorter_main_temp.py`** - Temporary backup file

### Test Categories

#### Functionality Tests
- `test_comprehensive_functions.py`
- `test_incremental_metadata.py` 
- `test_metadata_extraction.py`
- `test_folder_detection.py`

#### Data Handling Tests
- `test_unicode.py`
- `test_unicode_modes.py`
- `test_unicode_printing.py`
- `test_normalization.py`

#### JSON/Metadata Tests
- `test_malformed_json.py`
- `test_specific_json.py`
- `test_readme_parsing.py`
- `test_readme_filtering.py`

#### Performance Tests
- `test_simple_incremental.py`

## Usage

### Running Benchmark Tests
```bash
cd test-scripts
python3 hash_speed_test.py  # Automatically uses .safetensors files in directory
```

### Running Verification Tests
```bash
cd test-scripts
python3 test_autov3.py      # Tests AutoV3 hash on available .safetensors
python3 test_metadata.py   # Tests metadata extraction
```

### Adding New Test Files
Simply copy any `.safetensors` files to this directory and they will be automatically detected by the test scripts.

### Importing Main Scripts
All test scripts use `test_utils.py` to properly import from the parent directory:
```python
from test_utils import setup_parent_import
setup_parent_import()
from file_scanner import FileScanner
```

## Benefits

1. **Centralized Testing**: All test scripts in one location
2. **Automatic File Detection**: Scripts automatically find .safetensors files
3. **Consistent Interface**: All scripts use common utilities
4. **Easy Maintenance**: Clear separation of production vs test code
5. **Portable Tests**: Self-contained test environment

## Adding New Test Scripts

1. Create your test script in this directory
2. Import utilities: `from test_utils import setup_parent_import, get_safetensors_files`
3. Use `setup_parent_import()` before importing main modules
4. Use `get_safetensors_files()` to find test files
5. Follow the established patterns for consistency