# Server Deployment Guide: Column Lacking Error Resolution

## Issue
Your server is encountering errors like:
```
Warning: Could not save comprehensive metadata for /path/to/file.png: table media_metadata has no column named addnet_enabled
```

This happens because the server database was created before the enhanced metadata columns were added.

## Solution Steps

### Step 1: Update Database Schema
Run this on your server to add missing columns:

```bash
cd /path/to/Stable-model-manager
python update_database_schema.py model_sorter.sqlite
```

**Expected output:**
```
📋 Created backup: model_sorter.sqlite.backup_1760285878
Updating database schema: model_sorter.sqlite
✅ Added column: batch_size (INTEGER)
✅ Added column: batch_pos (INTEGER)  
✅ Added column: eta (REAL)
✅ Added column: ensd (INTEGER)
✅ Added column: face_restoration (TEXT)
✅ Added column: restore_faces (INTEGER)
✅ Added column: tiled_diffusion (TEXT)
✅ Added column: hires_resize_mode (TEXT)
✅ Added column: first_pass_size (TEXT)
✅ Added column: addnet_enabled (INTEGER)
✅ Added column: addnet_module_1 (TEXT)
✅ Added column: addnet_model_1 (TEXT)
✅ Added column: addnet_weight_1 (REAL)

🎉 Schema update complete: 13 columns added
```

### Step 2: Retry Previously Failed Files
After updating the schema, retry files that failed due to missing columns:

```bash
cd /path/to/Stable-model-manager
python model_sorter_main.py --retry-failed
```

**Expected output:**
```
Found X files that previously failed due to missing columns
Retrying metadata extraction...
✅ Successfully processed: filename1.png
✅ Successfully processed: filename2.jpeg
...
Retry complete: X processed, Y updated, 0 errors, Z skipped
```

### Step 3: Continue Normal Processing
Once the schema is updated and failed files retried, continue normal processing:

```bash
# Continue metadata extraction from where it left off
python model_sorter_main.py --extract-metadata-all-rescan
```

## Features Added

### 1. **Automatic Error Tracking**
- Failed files are automatically logged to `column_lacking.json`
- Tracks file path, column name, error message, and timestamp
- No duplicate entries for the same file

### 2. **Retry Functionality** 
- `--retry-failed` flag retries previously failed files
- Works with both `model_sorter_main.py` and `file_scanner.py`
- Updates retry count and timestamps

### 3. **Enhanced Metadata Columns**
The following columns are now supported:

**Generation Parameters:**
- `batch_size` - Batch size used in generation
- `batch_pos` - Position in batch  
- `eta` - ETA parameter
- `ensd` - Eta noise seed delta

**Face Restoration:**
- `face_restoration` - Face restoration method
- `restore_faces` - Whether faces were restored (boolean)

**Advanced Settings:**
- `tiled_diffusion` - Tiled diffusion settings
- `hires_resize_mode` - Highres resize mode
- `first_pass_size` - First pass size for highres

**AddNet/LoRA Parameters:**
- `addnet_enabled` - Whether AddNet is enabled
- `addnet_module_1` - First AddNet module name
- `addnet_model_1` - First AddNet model name  
- `addnet_weight_1` - First AddNet weight value

## Commands Summary

```bash
# Update database schema (run once)
python update_database_schema.py model_sorter.sqlite

# Retry previously failed files (after schema update)
python model_sorter_main.py --retry-failed

# Continue normal metadata extraction
python model_sorter_main.py --extract-metadata-all-rescan

# Check retry status
cat column_lacking.json | head -10
```

## Files Created

- `update_database_schema.py` - Database schema updater
- `column_lacking.json` - Tracks failed files for retry
- `model_sorter.sqlite.backup_*` - Automatic backup before schema changes

The system now handles missing database columns gracefully and provides tools to resolve the issue without data loss!