# Script Core Functionality Documentation

This document provides detailed information about each script's purpose, functionality, and relationships within the Stable Diffusion Model Organization System.

---

## Core Production Scripts (Root Directory)

### 1. `model_sorter_main.py` - **Main Entry Point & Pipeline Orchestrator** 
**Size**: 87,234 bytes | **Lines**: 1,726

**Purpose**: Central command hub that orchestrates the complete model organization pipeline from scanning to final organization.

**Key Functionality**:
- **Pipeline Orchestration**: Coordinates all processing steps (scan → metadata → duplicates → sort → civitai)
- **Command Processing**: Handles 15+ command-line operations with step-by-step execution
- **Configuration Management**: Loads and validates configuration from `config.ini`
- **Report Generation**: Creates comprehensive processing reports in `processing_report/` directory
- **Dry-run Support**: Previews all operations without making actual changes
- **Console Capture**: Captures and logs all output for review and debugging
- **Batch Processing**: Supports various batch sizes and commit strategies
- **Error Recovery**: Handles interruptions gracefully with proper database commits

**Dependencies**: 
- Imports: `FileScanner`, `DatabaseManager`, `MetadataExtractor`, `DuplicateDetector`, `ModelSorter`, `CivitaiInfoGenerator`
- Database: `model_sorter.sqlite`
- Config: `config.ini`

**Use Cases**:
- Complete model organization workflow
- Step-by-step processing (scan, metadata, duplicates, sort, civitai)
- Database maintenance operations
- System verification and reporting

---

### 2. `file_scanner.py` - **Core File Detection & Hash Calculation Engine**
**Size**: 315,172 bytes | **Lines**: 6,519

**Purpose**: Foundation module that scans directories, identifies model files, calculates hashes, and manages the database.

**Key Functionality**:
- **Multi-format Detection**: Recognizes 10+ model formats (.safetensors, .ckpt, .pt, .pth, .bin, .vae, .onnx, .gguf, .safe, .oldsafetensors)
- **Hash Calculation**: Supports SHA256 (universal), BLAKE3 (fast), and AutoV3 (SafeTensors-specific) hashing
- **Database Management**: Complete SQLite database operations via `DatabaseManager` class
- **Associated File Tracking**: Links preview images and metadata files to model files
- **Incremental Scanning**: Tracks scanned files to avoid redundant processing
- **Path Recovery**: Automatically recovers missing files by checking alternative paths
- **Metadata Extraction**: Direct SafeTensors header parsing for rapid metadata access
- **Cross-referencing**: Links files with Civitai database for enhanced model detection
- **Performance Optimization**: Batch processing, memory-efficient file reading, optimized database operations

**Classes**:
- `DatabaseManager`: SQLite database operations and schema management
- `FileScanner`: Main scanning and file detection logic

**Hash Strategy**:
- **SafeTensors**: AutoV3 (tensor data only, ~342 MB/s) + SHA256 (full file, ~577 MB/s)
- **Other files**: BLAKE3 (ultra-fast, ~3000+ MB/s) + SHA256 (compatibility, ~242 MB/s)

---

### 3. `metadata_extractor.py` - **Metadata Processing & Analysis Engine**
**Size**: 47,739 bytes | **Lines**: 1,025

**Purpose**: Extracts, parses, and standardizes metadata from various sources including SafeTensors headers, .metadata.json, and .civitai.info files.

**Key Functionality**:
- **SafeTensors Reader**: Direct header parsing without loading full model into memory
- **Multi-source Parsing**: Handles .metadata.json, .civitai.info, and embedded metadata
- **Metadata Standardization**: Normalizes metadata across different formats and sources
- **Batch Processing**: Efficiently processes multiple files with configurable limits
- **Error Recovery**: Handles malformed files and missing metadata gracefully
- **Database Integration**: Stores extracted metadata with proper indexing and relationships
- **Civitai Integration**: Cross-references with Civitai database for authoritative information
- **Format Validation**: Ensures metadata integrity and handles version differences

**Supported Formats**:
- SafeTensors embedded metadata (JSON in header)
- Standalone .metadata.json files
- Civitai .civitai.info files
- Legacy metadata formats

**Output**: Structured metadata entries in database with standardized fields

---

### 4. `duplicate_detector.py` - **Intelligent Duplicate Detection & Management**
**Size**: 22,223 bytes | **Lines**: 525

**Purpose**: Identifies duplicate model files using hash comparison and implements intelligent metadata preservation strategies.

**Key Functionality**:
- **Hash-based Detection**: Uses SHA256 hashes to identify identical files regardless of filename
- **Metadata Scoring**: Prioritizes files with better metadata (civitai.info > metadata.json > path depth)
- **Intelligent Preservation**: Keeps the "best" version of duplicates based on metadata quality
- **Duplicate Resolution**: Provides strategies for handling duplicates (keep best, move others, etc.)
- **Cross-reference Analysis**: Uses Civitai database to validate model information
- **Report Generation**: Detailed duplicate analysis reports with recommendations
- **Batch Processing**: Efficiently processes large datasets with optimized queries
- **Safety Checks**: Verifies file integrity before marking as duplicates

**Detection Algorithm**:
1. Group files by identical SHA256 hash
2. Score each file based on metadata availability and quality
3. Identify the highest-scoring file as the "keeper"
4. Flag others as duplicates with recommended actions
5. Generate detailed reports for manual review

---

### 5. `model_sorter.py` - **File Organization & Movement Engine**
**Size**: 63,184 bytes | **Lines**: 1,279

**Purpose**: Handles the physical organization of model files into structured directory hierarchies with safety checks and rollback capabilities.

**Key Functionality**:
- **Structured Organization**: Organizes models into `loras/base_model/model_name/` hierarchy
- **Associated File Management**: Moves preview images and metadata files alongside models
- **Safety Checks**: Verifies file integrity before and after moves
- **Collision Handling**: Manages filename conflicts with intelligent numbering
- **Rollback Support**: Can undo organization operations if issues occur
- **Dry-run Simulation**: Shows exactly what would be moved without making changes
- **Progress Tracking**: Reports progress during large organization operations
- **Directory Creation**: Automatically creates necessary directory structures

**Classes**:
- `FileMover`: Core file moving operations with safety checks
- `ModelSorter`: High-level organization logic and workflow management

**Organization Strategy**:
- Base model detection from metadata or Civitai database
- Model name extraction and sanitization  
- Duplicate handling with numbered directories
- Associated file linking and movement

---

### 6. `civitai_generator.py` - **Civitai Info File Generator & Metadata Enhancer**
**Size**: 28,119 bytes | **Lines**: 643

**Purpose**: Generates missing .civitai.info files from available metadata or Civitai database lookups to ensure complete model information.

**Key Functionality**:
- **Info File Generation**: Creates properly formatted .civitai.info files from existing metadata
- **Civitai Database Integration**: Looks up authoritative model information from civitai.sqlite
- **Metadata Enhancement**: Adds missing model information (type, base model, tags, etc.)
- **Template System**: Uses standardized templates for consistent info file formatting
- **Batch Processing**: Efficiently processes multiple models with database optimization
- **Quality Validation**: Ensures generated files meet Civitai info standards
- **Missing Data Handling**: Gracefully handles models not found in Civitai database
- **Preview Integration**: Links preview images and additional resources

**Generated Content**:
- Model name, type, and base model information
- Creator/author details and model description
- Tags, categories, and version information
- Download URLs and model statistics
- Associated preview images and resources

---

## Auxiliary Scripts (test-scripts/ Directory)

### Utility Scripts (Maintenance & Operations)

#### `add_missing_metadata_files.py`
**Purpose**: Database maintenance utility that adds missing .metadata.json entries to the scanned_files table
**Use Case**: Recovery after metadata files were created post-scan
**Functionality**: Scans for orphaned metadata files and adds database entries

#### `fix_database_paths.py`  
**Purpose**: Repairs database path inconsistencies and broken file references
**Use Case**: After moving files or changing directory structures
**Functionality**: Updates database paths to match current file locations

#### `fix_paths_to_user.py`
**Purpose**: Migrates database paths from old to new directory prefixes
**Use Case**: System migration or directory restructuring
**Functionality**: Batch updates paths (e.g., '/old/path/' to '/new/path/')

#### `update_database_schema.py`
**Purpose**: Updates database schema to newer versions
**Use Case**: System upgrades requiring database structure changes
**Functionality**: Adds columns, indexes, and tables as needed

#### `view_report.py`
**Purpose**: Interactive viewer for processing reports and statistics
**Use Case**: Analysis of processing results and system performance
**Functionality**: Displays formatted reports with filtering and search

### Performance & Debug Scripts

#### `hash_speed_test.py`
**Purpose**: Benchmarks hash calculation performance across different algorithms
**Use Case**: Performance optimization and algorithm comparison
**Functionality**: Tests SHA256, BLAKE3, and AutoV3 with various chunk sizes

#### `debug_cifs_performance.py`
**Purpose**: Diagnoses network file system performance issues
**Use Case**: Troubleshooting slow processing on network drives
**Functionality**: Isolates file operation bottlenecks and suggests optimizations

#### `fast_prescan_optimization.py`
**Purpose**: Tests prescan performance optimizations
**Use Case**: Improving initial directory scanning speed
**Functionality**: Compares different scanning strategies and caching methods

#### `optimized_db_lookup.py`
**Purpose**: Performance testing for database lookup operations
**Use Case**: Optimizing query performance for large datasets
**Functionality**: Benchmarks different query strategies and indexing approaches

### Verification & Testing Scripts

#### `feature_verification.py`
**Purpose**: Comprehensive testing suite for all system features
**Use Case**: System validation and regression testing
**Functionality**: Tests all major features with various scenarios

#### `final_verification.py`
**Purpose**: Final system health check and validation
**Use Case**: Pre-deployment verification or post-processing validation
**Functionality**: Verifies database integrity, file consistency, and system state

#### `test_autov3.py`
**Purpose**: Specialized testing for AutoV3 hash computation
**Use Case**: Validating SafeTensors hash calculation accuracy
**Functionality**: Tests AutoV3 against known good hashes

### Demo & Documentation Scripts

#### `demo_cross_reference.py`
**Purpose**: Interactive demonstration of cross-referencing capabilities
**Use Case**: Showcasing system features to users
**Functionality**: Walks through cross-referencing process with examples

#### `demo_incremental_complete.py`
**Purpose**: Demonstrates incremental processing workflow
**Use Case**: Training and system overview
**Functionality**: Shows step-by-step incremental processing

#### `metadata_extractor_cli.py`
**Purpose**: Command-line interface wrapper for metadata extraction
**Use Case**: Standalone metadata extraction without full pipeline
**Functionality**: Extracts metadata from individual files or directories

#### `blake3_integration_docs.py`
**Purpose**: Living documentation for BLAKE3 hash integration
**Use Case**: Developer reference and implementation guide
**Functionality**: Documents BLAKE3 usage patterns and performance characteristics

### Backup & Legacy Files

#### `model_sorter_main_backup.py`
**Purpose**: Backup copy of main script before major changes
**Use Case**: Rollback capability during development
**Functionality**: Previous working version for emergency restoration

#### `model_sorter_main_temp.py`
**Purpose**: Temporary backup during development
**Use Case**: Development safety net
**Functionality**: Working copy during experimental changes

---

## Integration & Dependencies

### Database Dependencies
- **Primary**: `model_sorter.sqlite` (main application database)
- **Secondary**: `Database/civitai.sqlite` (reference data for model information)

### Configuration Dependencies  
- **Primary**: `config.ini` (main configuration file)
- **Sections**: Database paths, processing options, performance tuning, output directories

### Import Hierarchy
```
model_sorter_main.py
├── file_scanner.py (FileScanner, DatabaseManager)
├── metadata_extractor.py (MetadataExtractor)  
├── duplicate_detector.py (DuplicateDetector)
├── model_sorter.py (ModelSorter)
└── civitai_generator.py (CivitaiInfoGenerator)
```

### Processing Pipeline Flow
1. **Scan**: `FileScanner` discovers and hashes model files
2. **Extract**: `MetadataExtractor` processes embedded and external metadata
3. **Detect**: `DuplicateDetector` identifies and resolves duplicate files
4. **Sort**: `ModelSorter` organizes files into structured directories
5. **Generate**: `CivitaiInfoGenerator` creates missing info files
6. **Report**: `model_sorter_main.py` generates comprehensive reports

---

## Usage Patterns & Best Practices

### Development Workflow
1. Use auxiliary scripts in `test-scripts/` for testing and verification
2. Keep core scripts in root for production operations
3. Use `--dry-run` extensively before actual operations
4. Monitor processing reports for issues and optimization opportunities

### Performance Optimization
- Use `hash_speed_test.py` to optimize hash calculation settings
- Use `debug_cifs_performance.py` for network storage optimization
- Monitor batch sizes and commit frequencies for optimal performance

### Maintenance Operations
- Use `fix_database_paths.py` after directory restructuring
- Use `update_database_schema.py` for system upgrades
- Use verification scripts before and after major operations

This documentation serves as the definitive reference for understanding the purpose and capabilities of each script in the system.