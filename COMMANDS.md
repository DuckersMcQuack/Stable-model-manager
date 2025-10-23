# Model Sorter Commands Reference

This document provides a comprehensive reference of all available commands for the Model Sorter project scripts.

## Table of Contents
1. [Main Pipeline (`model_sorter_main.py`)](#main-pipeline-model_sorter_mainpy)
2. [File Scanner (`file_scanner.py`)](#file-scanner-file_scannerpy)
3. [Model Sorter (`model_sorter.py`)](#model-sorter-model_sorterpy)
4. [Utility Scripts](#utility-scripts)
5. [Common Workflows](#common-workflows)

---

## Main Pipeline (`model_sorter_main.py`)

### Overview
Unified pipeline for comprehensive model organization with duplicate detection, metadata extraction, and sorting.

### Basic Commands

#### Complete Workflow
```bash
# Full organization workflow (preview first!)
python model_sorter_main.py --dry-run

# Execute full workflow
python model_sorter_main.py

# Use custom configuration
python model_sorter_main.py --config custom_config.ini
```

### Step-by-Step Execution

#### Individual Steps
```bash
# Step 1: Scan source directories
python model_sorter_main.py --step scan

# Step 2: Extract metadata from model files
python model_sorter_main.py --step metadata

# Step 3: Detect and resolve duplicates
python model_sorter_main.py --step duplicates

# Step 4: Sort and organize models
python model_sorter_main.py --step sort

# Step 5: Generate civitai.info files
python model_sorter_main.py --step civitai

# Rescan for new files
python model_sorter_main.py --step rescan

# Extract metadata from media files
python model_sorter_main.py --step extract-metadata

# Retry failed operations
python model_sorter_main.py --step retry-failed

# Migrate database paths
python model_sorter_main.py --step migrate-paths

# Recover missing files
python model_sorter_main.py --step recover-missing

# Force BLAKE3 rehashing
python model_sorter_main.py --step force-blake3

# Update database tables
python model_sorter_main.py --step update-tables

# Rescan and verify file associations
python model_sorter_main.py --step rescan-linked

# Rebuild file associations from scratch
python model_sorter_main.py --step rescan-linked-rebuild

# Repair metadata/text file paths
python model_sorter_main.py --step rescan-and-repair-metadata-text
```

### Advanced Options

#### Scanning Options
```bash
# Force rescan all files (ignores cache)
python model_sorter_main.py --step scan --force-rescan

# Control folder processing
python model_sorter_main.py --skip-folders          # Skip folder content scanning (default)
python model_sorter_main.py --no-skip-folders       # Include folder content scanning
python model_sorter_main.py --folder-limit 100      # Limit folders per batch
```

#### Metadata Processing
```bash
# Extract metadata from limited number of files
python model_sorter_main.py --extract-metadata-limit 500

# Extract metadata from ALL media files
python model_sorter_main.py --extract-metadata-all

# Extract metadata with forced rescan
python model_sorter_main.py --extract-metadata-all-rescan
```

#### Database Operations
```bash
# Migrate database paths (OLD_PREFIX to NEW_PREFIX)
python model_sorter_main.py --migrate-paths "/old/path/" "/new/path/"

# Migrate data between database tables
python model_sorter_main.py --migrate-tables

# Update database schema
python model_sorter_main.py --update-tables

# Recover missing files automatically
python model_sorter_main.py --recover-missing

# Retry failed operations
python model_sorter_main.py --retry-failed
```

#### File Association Management
```bash
# Verify and correct file associations (recommended)
python model_sorter_main.py --rescan-linked

# With detailed verbose output showing each model and file processed
python model_sorter_main.py --rescan-linked --verbose

# Completely rebuild all associations (use if corrupted)
python model_sorter_main.py --rescan-linked-rebuild

# With verbose output for complete rebuild
python model_sorter_main.py --rescan-linked-rebuild --verbose

# Repair metadata/text file path mismatches
python model_sorter_main.py --rescan-and-repair-metadata-text
```

#### Hashing Options
```bash
# Force BLAKE3 rehashing for better performance
python model_sorter_main.py --force-blake3
```

#### Output Control
```bash
# Quiet mode (minimal output)
python model_sorter_main.py --quiet

# Always use with --dry-run first!
python model_sorter_main.py --step sort --dry-run
```

---

## File Scanner (`file_scanner.py`)

### Overview
Core scanning component for file hashing, metadata extraction, and model enhancement.

### Basic Scanning
```bash
# Scan a directory for models and media
python file_scanner.py --scan-directory /path/to/models

# Scan with forced rescan (ignore cache)
python file_scanner.py --scan-directory /path/to/models --force-rescan

# Preview scanning without changes
python file_scanner.py --scan-directory /path/to/models --dry-run

# Use custom configuration
python file_scanner.py --scan-directory /path/to/models --config custom_config.ini
```

### Model Organization
```bash
# Sort models into organized structure
python file_scanner.py --sort-models

# Preview sorting without moving files
python file_scanner.py --sort-models --dry-run
```

### Model Enhancement
```bash
# Enhance model records with civitai data and pattern detection
python file_scanner.py --enhance-models

# Enhanced verbose output
python file_scanner.py --enhance-models --verbose

# Directly enhance Unknown/Other models from server database
python file_scanner.py --enhance-models-direct

# Enhance with custom database path
python file_scanner.py --enhance-models-direct /path/to/database.sqlite

# Apply enhancements to database (use with --enhance-models-direct)
python file_scanner.py --enhance-models-direct --apply-enhancements

# Verbose enhancement with database updates
python file_scanner.py --enhance-models-direct --apply-enhancements --verbose
```

### Metadata Processing
```bash
# Extract and analyze metadata from media files
python file_scanner.py --extract-metadata

# Retry failed metadata extractions
python file_scanner.py --retry-failed

# Process orphaned media files using cross-referencing
python file_scanner.py --process-orphaned
```

### Database Operations
```bash
# Migrate database paths
python file_scanner.py --migrate-paths "/old/path/" "/new/path/"

# Preview path migration
python file_scanner.py --migrate-paths "/old/path/" "/new/path/" --dry-run-migrate

# Recover missing files automatically
python file_scanner.py --recover-missing

# Repair text/metadata file path mismatches
python file_scanner.py --repair-text-paths
```

### Verbose Output
```bash
# Enable detailed logging for any command
python file_scanner.py --enhance-models-direct --verbose
python file_scanner.py --scan-directory /path/to/models --verbose
```

---

## Model Sorter (`model_sorter.py`)

### Overview
Standalone model sorting with civitai integration.

### Basic Usage
```bash
# Sort models with preview
python model_sorter.py --dry-run

# Execute sorting
python model_sorter.py

# Use custom configuration
python model_sorter.py --config custom_config.ini
```

---

## Utility Scripts

### Add Missing Metadata Files (`add_missing_metadata_files.py`)
```bash
# Preview missing metadata files (dry run)
python add_missing_metadata_files.py

# Execute addition of missing metadata files
python add_missing_metadata_files.py --execute

# Use custom database
python add_missing_metadata_files.py --database /path/to/database.sqlite --execute
```

---

## Common Workflows

### Complete Model Organization Workflow
```bash
# 1. Preview the complete workflow first
python model_sorter_main.py --dry-run

# 2. Execute the full pipeline
python model_sorter_main.py

# 3. Enhance model information
python file_scanner.py --enhance-models --verbose

# 4. Process any Unknown/Other models
python file_scanner.py --enhance-models-direct --apply-enhancements --verbose
```

### Initial Setup and Scanning
```bash
# 1. Scan source directories
python file_scanner.py --scan-directory /path/to/models --verbose

# 2. Extract metadata
python model_sorter_main.py --step metadata

# 3. Enhance model information
python file_scanner.py --enhance-models --verbose

# 4. Preview sorting
python model_sorter_main.py --step sort --dry-run

# 5. Execute sorting
python model_sorter_main.py --step sort
```

### Database Maintenance
```bash
# 1. Verify file associations
python model_sorter_main.py --rescan-linked

# 2. Repair path mismatches
python file_scanner.py --repair-text-paths

# 3. Recover missing files
python file_scanner.py --recover-missing

# 4. Update database schema if needed
python model_sorter_main.py --update-tables
```

### Model Enhancement Workflow
```bash
# 1. Enhance existing models
python file_scanner.py --enhance-models --verbose

# 2. Target Unknown/Other models specifically
python file_scanner.py --enhance-models-direct --apply-enhancements --verbose

# 3. Verify results
python file_scanner.py --enhance-models --verbose
```

### Path Migration
```bash
# 1. Preview path migration
python file_scanner.py --migrate-paths "/old/path/" "/new/path/" --dry-run-migrate

# 2. Execute path migration
python file_scanner.py --migrate-paths "/old/path/" "/new/path/"

# 3. Verify and repair any issues
python model_sorter_main.py --rescan-linked
```

### Performance Optimization
```bash
# Use BLAKE3 for faster hashing
python model_sorter_main.py --force-blake3

# Process in smaller batches
python model_sorter_main.py --folder-limit 50

# Use quiet mode for less output
python model_sorter_main.py --quiet
```

---

## Important Notes

### Safety First
- **Always use `--dry-run` first** to preview changes before executing
- The scripts will move and organize files - make sure you have backups
- Reports are saved to `processing_report/` directory for review

### Database Requirements
- Requires `civitai.sqlite` database in `Database/` directory for enhanced model detection
- Local `model_sorter.sqlite` database is created automatically for tracking

### Configuration
- Default configuration is `config.ini`
- Use `--config` to specify custom configuration files
- Most commands support verbose output with `--verbose`

### File Types Supported
- **Model files**: `.safetensors`, `.ckpt`, `.pt`, `.pth`, `.bin`, `.onnx`, `.gguf`
- **Metadata files**: `.json`, `.yaml`, `.txt`, `.civitai.info`
- **Media files**: `.png`, `.jpg`, `.jpeg`, `.webp`, `.gif`

### Enhancement Features
- **Civitai integration**: Automatic model name and metadata detection
- **Pattern detection**: Intelligent base model detection from filenames and paths
- **Associated file analysis**: Checks metadata files for model information
- **Duplicate handling**: SHA256-based duplicate detection and resolution