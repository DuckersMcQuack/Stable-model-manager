#!/usr/bin/env python3
"""
Model Sorter - Stable Diffusion Model Organization Pipeline

Unified pipeline that:
 - Recursively scans source directories for model files (.safetensors, .ckpt, .pt)
 - Calculates SHA256 hashes for duplicate detection across different file names
 - Extracts metadata from SafeTensors headers, .metadata.json, .civitai.info files
 - Identifies and resolves duplicates using scoring system (civitai.info > metadata.json > file path depth)
 - Organizes models into structured directories: loras/base_model/model_name/
 - Generates missing civitai.info files from metadata or civitai database lookups
 - Integrates with civitai.sqlite database for authoritative model type/base model detection
 - Supports dry-run mode with comprehensive reporting and console output capture
 - Handles associated files (previews, metadata, etc.) alongside model files
 - Never skips files - duplicates are moved to numbered subdirectories

Notes:
 - This script will move/organize files. Use --dry-run to preview actions first.
 - Reports are saved to processing_report/ directory for review and verification.
 - Requires civitai.sqlite database in Database/ for enhanced model detection.

Usage examples:
  python model_sorter_main.py --dry-run                    # Preview what would be done
  python model_sorter_main.py                              # Run full organization workflow  
  python model_sorter_main.py --step scan                  # Run only file scanning
  python model_sorter_main.py --step sort --dry-run        # Preview only the sorting step
  python model_sorter_main.py --config custom_config.ini   # Use custom configuration
"""

import argparse
import configparser
import json
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from io import StringIO
from typing import Optional

# Import our modules
from file_scanner import FileScanner, DatabaseManager
from metadata_extractor import MetadataExtractor
from duplicate_detector import DuplicateDetector
from model_sorter import ModelSorter
from civitai_generator import CivitaiInfoGenerator


class ConsoleCapture:
    """Capture console output for inclusion in reports"""
    
    def __init__(self):
        self.captured_output = []
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
    def start_capture(self):
        """Start capturing console output"""
        self.capture_buffer = StringIO()
        sys.stdout = TeeOutput(self.original_stdout, self.capture_buffer)
        sys.stderr = TeeOutput(self.original_stderr, self.capture_buffer)
        
    def stop_capture(self):
        """Stop capturing and return captured output"""
        if hasattr(self, 'capture_buffer'):
            sys.stdout = self.original_stdout  
            sys.stderr = self.original_stderr
            captured = self.capture_buffer.getvalue()
            self.capture_buffer.close()
            return captured.splitlines()
        return []


class TeeOutput:
    """Tee output to both original stream and capture buffer"""
    
    def __init__(self, original, capture):
        self.original = original
        self.capture = capture
        
    def write(self, text):
        self.original.write(text)
        self.capture.write(text)
        
    def flush(self):
        self.original.flush()
        self.capture.flush()


class ModelSorterOrchestrator:
    """Main orchestrator class that coordinates all components"""
    
    def __init__(self, config_path: str = "config.ini", dry_run: bool = False, 
                 verbose: bool = True, force_rescan: bool = False, extract_metadata_limit: Optional[int] = None,
                 extract_metadata_all: bool = False, extract_metadata_all_rescan: bool = False, retry_failed: bool = False,
                 skip_folders: bool = True, folder_limit: Optional[int] = None):
        self.config_path = config_path
        self.dry_run = dry_run
        self.verbose = verbose
        self.force_rescan = force_rescan
        self.extract_metadata_limit = extract_metadata_limit or 100  # Default to 100 if not specified
        self.extract_metadata_all = extract_metadata_all
        self.extract_metadata_all_rescan = extract_metadata_all_rescan
        self.retry_failed = retry_failed
        self.skip_folders = skip_folders
        self.folder_limit = folder_limit
        
        # Initialize console capture
        self.console_capture = ConsoleCapture()
        
        # Initialize components
        self.scanner = FileScanner(config_path, force_rescan=force_rescan, skip_folders=skip_folders, verbose=verbose)
        self.metadata_extractor = MetadataExtractor()
        self.duplicate_detector = DuplicateDetector()
        self.model_sorter = ModelSorter(config_path, dry_run)
        self.civitai_generator = CivitaiInfoGenerator()
        
        # Stats
        self.start_time = time.time()
        self.stats = {
            'files_scanned': 0,
            'models_found': 0,
            'duplicates_found': 0,
            'models_sorted': 0,
            'models_migrated': 0,
            'civitai_files_generated': 0,
            'errors': 0
        }
    
    def print_step(self, step_name: str, description: str = ""):
        """Print a step header"""
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"STEP: {step_name}")
            if description:
                print(f"Description: {description}")
            print(f"{'='*60}")
    
    def print_progress(self, message: str):
        """Print progress message"""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {message}")
    
    def load_config(self):
        """Load configuration from config file"""
        import configparser
        config = configparser.ConfigParser()
        config.read(self.config_path)
        return config
    
    def get_server_database_path(self) -> str:
        """Get server database path from config"""
        config = self.load_config()
        try:
            return config.get('Paths', 'server_database_path')
        except (configparser.NoSectionError, configparser.NoOptionError):
            # Fallback to default if not configured
            return "/path/to/Stable-model-manager/model_sorter.sqlite"
    
    def get_destination_directory(self) -> str:
        """Get destination directory path from config"""
        config = self.load_config()
        try:
            return config.get('Paths', 'destination_directory')
        except (configparser.NoSectionError, configparser.NoOptionError):
            # Fallback to empty string if not configured
            return ""

    def step_1_scan_files(self) -> bool:
        """Step 1: Scan source directory for files and generate hashes"""
        try:
            # Detect if this is a fast prescan (dry-run mode for preliminary mapping)
            if self.dry_run:
                self.print_step("1. ENHANCED PRESCAN (DRY-RUN)", 
                              "Check all files, add missing ones to database with full metadata extraction")
                fast_prescan = True
            else:
                self.print_step("1. FILE SCANNING", 
                              "Scanning source directory and calculating SHA256 hashes")
                fast_prescan = False
            
            source_dir = self.scanner.config['source_directory']
            self.print_progress(f"Scanning directory: {source_dir}")
            
            if not os.path.exists(source_dir):
                print(f"ERROR: Source directory not found: {source_dir}")
                return False
            
            # Perform the scan (with fast prescan mode if dry-run and folder limit if specified)
            results = self.scanner.scan_directory(source_dir, fast_prescan=fast_prescan, force_blake3=self.force_rescan, folder_limit=self.folder_limit, verbose_dry_run=(self.dry_run or self.verbose))
            
            # Update stats
            self.stats['files_scanned'] = sum(len(files) for files in results.values())
            self.stats['models_found'] = len(results['models'])
            
            if fast_prescan:
                self.print_progress(f"Fast prescan complete: {self.stats['files_scanned']} unscanned files identified")
                self.print_progress(f"SafeTensors models needing full scan: {self.stats['models_found']}")
                self.print_progress("Use --step scan (without --dry-run) to perform full scanning")
            else:
                self.print_progress(f"Scan complete: {self.stats['files_scanned']} files processed")
                self.print_progress(f"Model files found: {self.stats['models_found']}")
            
            return True
            
        except Exception as e:
            print(f"ERROR in file scanning: {e}")
            self.stats['errors'] += 1
            return False
    
    def step_2_extract_metadata(self) -> bool:
        """Step 2: Extract metadata from model files"""
        try:
            self.print_step("2. METADATA EXTRACTION", 
                          "Extracting metadata from model files and associated text files")
            
            self.print_progress("Processing scanned model files...")
            self.metadata_extractor.process_scanned_models()
            
            self.print_progress("Metadata extraction complete")
            return True
            
        except Exception as e:
            print(f"ERROR in metadata extraction: {e}")
            self.stats['errors'] += 1
            return False
    
    def step_3_detect_duplicates(self) -> bool:
        """Step 3: Detect and handle duplicates"""
        try:
            self.print_step("3. DUPLICATE DETECTION", 
                          "Identifying duplicates and determining best versions to keep")
            
            self.print_progress("Analyzing files for duplicates...")
            duplicate_groups = self.duplicate_detector.create_duplicate_groups()
            
            summary = self.duplicate_detector.get_duplicate_summary()
            self.stats['duplicates_found'] = summary['duplicate_files']
            
            self.print_progress(f"Duplicate detection complete")
            self.print_progress(f"Duplicate groups: {summary['duplicate_groups']}")
            self.print_progress(f"Duplicate files: {summary['duplicate_files']}")
            
            return True
            
        except Exception as e:
            print(f"ERROR in duplicate detection: {e}")
            self.stats['errors'] += 1
            return False
    
    def step_4_sort_models(self) -> bool:
        """Step 4: Sort and move model files using batch processing"""
        try:
            self.print_step("4. MODEL SORTING", 
                          "Moving models to organized directory structure using batch processing")
            
            if self.dry_run:
                self.print_progress("DRY RUN MODE - No files will actually be moved")
            
            # Use the new batch processing system from FileScanner
            self.print_progress("Sorting models using batch processing (batch size from config.ini)...")
            sort_results = self.scanner.sort_models_batch()
            
            # Update stats from batch results
            self.stats['models_sorted'] = sort_results.get('models_moved', 0)
            
            self.print_progress(f"Batch sorting complete")
            self.print_progress(f"Total models processed: {sort_results.get('total_models_processed', 0)}")
            self.print_progress(f"Models moved: {sort_results.get('models_moved', 0)}")
            self.print_progress(f"Associated files moved: {sort_results.get('associated_files_moved', 0)}")
            self.print_progress(f"Database batches committed: {sort_results.get('batches_committed', 0)}")
            if sort_results.get('errors', 0) > 0:
                self.print_progress(f"Errors encountered: {sort_results.get('errors', 0)}")
            
            return True
            
        except Exception as e:
            print(f"ERROR in model sorting: {e}")
            self.stats['errors'] += 1
            return False
    
    def step_5_generate_civitai_files(self) -> bool:
        """Step 5: Generate missing civitai.info files"""
        try:
            self.print_step("5. CIVITAI.INFO GENERATION", 
                          "Generating civitai.info files for models that lack them")
            
            if self.dry_run:
                self.print_progress("DRY RUN MODE - No civitai.info files will be generated")
                return True
            
            self.print_progress("Generating missing civitai.info files...")
            success_count, error_count = self.civitai_generator.process_models_needing_civitai_info()
            
            # Also generate for moved models
            self.civitai_generator.generate_missing_civitai_files_for_moved_models()
            
            self.stats['civitai_files_generated'] = success_count
            self.stats['errors'] += error_count
            
            self.print_progress(f"Civitai.info generation complete")
            self.print_progress(f"Files generated: {success_count}")
            if error_count > 0:
                self.print_progress(f"Errors: {error_count}")
            
            return True
            
        except Exception as e:
            print(f"ERROR in civitai.info generation: {e}")
            self.stats['errors'] += 1
            return False
    
    def step_rescan_metadata(self) -> bool:
        """Rescan step: Update missing image metadata in database"""
        try:
            self.print_step("RESCAN", 
                          "Re-scanning files to update missing image metadata in database")
            
            self.print_progress("Checking for files with missing image metadata...")
            update_stats = self.scanner.rescan_missing_metadata()
            
            self.stats['files_updated'] = update_stats['updated']
            
            self.print_progress(f"Image metadata rescan complete")
            self.print_progress(f"Files updated: {update_stats['updated']}")
            
            return True
            
        except Exception as e:
            print(f"ERROR in metadata rescan: {e}")
            self.stats['errors'] += 1
            return False
    
    def step_extract_civitai_metadata(self) -> bool:
        """Extract enhanced Civitai metadata from existing files in database"""
        try:
            self.print_step("EXTRACT CIVITAI METADATA", 
                          "Scanning existing files to extract enhanced Civitai metadata")
            
            if self.extract_metadata_all_rescan:
                self.print_progress(f"Extracting Civitai metadata from ALL files (including already scanned)...")
                stats = self.scanner.scan_existing_files_for_metadata(None, force_rescan=True)
            elif self.extract_metadata_all:
                self.print_progress(f"Extracting Civitai metadata from ALL files needing processing...")
                stats = self.scanner.scan_existing_files_for_metadata(None)
            else:
                self.print_progress(f"Extracting Civitai metadata from up to {self.extract_metadata_limit} files...")
                stats = self.scanner.scan_existing_files_for_metadata(self.extract_metadata_limit)
            
            self.stats.update(stats)
            
            self.print_progress(f"Civitai metadata extraction complete")
            self.print_progress(f"Files processed: {stats['processed']}")
            self.print_progress(f"Files updated: {stats['updated']}")
            self.print_progress(f"Files skipped: {stats['skipped']}")
            self.print_progress(f"Errors: {stats['errors']}")
            
            return True
            
        except Exception as e:
            print(f"ERROR in Civitai metadata extraction: {e}")
            self.stats['errors'] += 1
            return False
    
    def step_retry_failed_metadata(self) -> bool:
        """Step: Retry metadata extraction for previously failed files"""
        try:
            self.print_step("RETRY FAILED METADATA EXTRACTION", 
                           "Retrying files that previously failed due to missing database columns")
            
            self.print_progress(f"Retrying metadata extraction for previously failed files...")
            stats = self.scanner.retry_failed_metadata_extraction()
            
            self.print_progress(f"Retry metadata extraction complete")
            self.print_progress(f"Files processed: {stats['processed']}")
            self.print_progress(f"Files updated: {stats['updated']}")
            self.print_progress(f"Files skipped: {stats['skipped']}")
            self.print_progress(f"Errors: {stats['errors']}")
            
            return True
            
        except Exception as e:
            print(f"ERROR in retry failed metadata extraction: {e}")
            self.stats['errors'] += 1
            return False
    
    def step_migrate_paths(self, old_prefix: str, new_prefix: str) -> bool:
        """Step: Migrate database paths from old prefix to new prefix"""
        try:
            self.print_step("MIGRATE DATABASE PATHS", 
                           f"Migrating paths from '{old_prefix}' to '{new_prefix}'")
            
            self.print_progress(f"Migrating database paths...")
            results = self.scanner.migrate_database_paths(old_prefix, new_prefix, dry_run=self.dry_run)
            
            self.print_progress(f"Path migration complete")
            self.print_progress(f"Files found with old prefix: {results['files_found']}")
            self.print_progress(f"Files verified at new path: {results['files_verified']}")
            self.print_progress(f"Files missing at new path: {results['files_missing']}")
            
            if not self.dry_run:
                self.print_progress(f"Files updated in database: {results['files_updated']}")
            else:
                self.print_progress(f"[DRY RUN] Would update {results['files_verified']} file paths")
            
            if results['files_missing'] > 0:
                self.print_progress(f"WARNING: {results['files_missing']} files not found at new paths")
                self.print_progress("Consider checking if files were actually moved/renamed")
            
            return True
            
        except Exception as e:
            print(f"ERROR in path migration: {e}")
            self.stats['errors'] += 1
            return False
    
    def step_recover_missing_files(self) -> bool:
        """Step: Recover missing files using automatic recovery strategies"""
        try:
            self.print_step("RECOVER MISSING FILES", 
                           "Automatically recovering missing files using alternative paths and destinations")
            
            self.print_progress(f"Scanning for missing files and attempting recovery...")
            results = self.scanner.bulk_recover_missing_files(dry_run=self.dry_run, limit=200)
            
            self.print_progress(f"File recovery complete")
            self.print_progress(f"Files checked: {results['total_checked']}")
            self.print_progress(f"Files recovered: {results['recovered']}")
            self.print_progress(f"Files still missing: {results['still_missing']}")
            
            if not self.dry_run:
                self.print_progress(f"Database updates: {results['db_updates']}")
            else:
                self.print_progress(f"[DRY RUN] Would update {results['recovered']} file paths")
            
            if results['recovered'] > 0:
                self.print_progress(f"Successfully recovered {results['recovered']} missing files!")
                
            if results['still_missing'] > 0:
                self.print_progress(f"WARNING: {results['still_missing']} files could not be recovered")
            
            return True
            
        except Exception as e:
            print(f"ERROR in file recovery: {e}")
            self.stats['errors'] += 1
            return False
    
    def step_force_blake3_rescan(self) -> bool:
        """Step: Force rescan of non-SafeTensors files to add BLAKE3 hashes"""
        try:
            self.print_step("FORCE BLAKE3 RESCAN", 
                           "Rescanning non-SafeTensors files to add fast BLAKE3 hashes")
            
            source_dir = self.scanner.config['source_directory']
            self.print_progress(f"Scanning directory for non-SafeTensors files: {source_dir}")
            
            if not os.path.exists(source_dir):
                print(f"ERROR: Source directory not found: {source_dir}")
                return False
            
            # Perform BLAKE3 rescan (force_blake3=True for non-SafeTensors)
            self.scanner.force_rescan = True  # Force rescanning
            results = self.scanner.scan_directory(source_dir, fast_prescan=False, force_blake3=True)
            
            # Update stats
            self.stats['files_scanned'] = sum(len(files) for files in results.values())
            self.stats['models_found'] = len(results['models'])
            
            self.print_progress(f"BLAKE3 rescan complete: {self.stats['files_scanned']} files processed")
            self.print_progress(f"Files updated with BLAKE3 hashes: {self.stats['models_found']}")
            self.print_progress("Non-SafeTensors files now have fast BLAKE3 hashes for future operations")
            
            return True
            
        except Exception as e:
            print(f"ERROR in BLAKE3 force rescan: {e}")
            self.stats['errors'] += 1
            return False
    
    def step_update_tables(self) -> bool:
        """Step: Check database completeness and rescan files with missing table entries"""
        try:
            self.print_step("UPDATE TABLES", 
                           "Checking database completeness and rescanning files with missing entries")
            
            source_dir = self.scanner.config['source_directory']
            self.print_progress(f"Analyzing database completeness for: {source_dir}")
            
            if not os.path.exists(source_dir):
                print(f"ERROR: Source directory not found: {source_dir}")
                return False
            
            # Get files that need rescanning due to incomplete database entries
            incomplete_files = self.scanner.db.get_files_needing_rescan()
            
            if not incomplete_files:
                self.print_progress("✓ All files in database have complete table entries")
                return True
            
            self.print_progress(f"Found {len(incomplete_files)} files with incomplete table entries")
            
            # Rescan incomplete files using existing metadata scanning functionality
            scan_results = self.scanner.scan_existing_files_for_metadata(
                limit=None,  # No limit, scan all incomplete files
                force_rescan=True
            )
            
            # Update stats
            self.stats['files_scanned'] = scan_results.get('processed', 0)
            
            self.print_progress(f"Database update complete:")
            self.print_progress(f"  - Files processed: {scan_results.get('processed', 0)}")
            self.print_progress(f"  - Files updated: {scan_results.get('updated', 0)}")
            self.print_progress(f"  - Files skipped: {scan_results.get('skipped', 0)}")
            self.print_progress(f"  - Errors encountered: {scan_results.get('errors', 0)}")
            
            if scan_results.get('errors', 0) > 0:
                self.stats['errors'] += scan_results['errors']
                self.print_progress("Some files had errors during rescanning")
            else:
                self.print_progress("All incomplete files have been successfully rescanned")
            
            return scan_results.get('errors', 0) == 0
            
        except Exception as e:
            print(f"ERROR in database update: {e}")
            self.stats['errors'] += 1
            return False
    
    def step_rescan_linked_files(self) -> bool:
        """Step: Verify and correct file associations (incremental check, not complete rebuild)"""
        try:
            self.print_step("VERIFY LINKED FILES", 
                           "Verifying and correcting file associations without complete rebuild")
            
            # Use server database from config for this operation
            config = self.load_config()
            server_db_path = config.get('Paths', 'server_database_path', fallback='model_sorter.sqlite')
            
            # Create scanner instance with server database for correlation logic
            from file_scanner import DatabaseManager, FileScanner
            import json
            import re
            temp_scanner = FileScanner(self.config_path)
            temp_scanner.db = DatabaseManager(server_db_path)
            temp_scanner.config['verbose'] = self.verbose
            conn = temp_scanner.db.conn
            
            if self.verbose:
                print(f"🔗 Using configured database: {server_db_path}")
            
            cursor = conn.cursor()
            
            # Get models that have metadata but missing associations
            cursor.execute('''
                SELECT mf.id, mf.scanned_file_id, sf.file_path, sf.file_name
                FROM model_files mf
                JOIN scanned_files sf ON mf.scanned_file_id = sf.id
                LEFT JOIN (
                    SELECT model_file_id, COUNT(*) as assoc_count
                    FROM associated_files
                    GROUP BY model_file_id
                ) af ON mf.id = af.model_file_id
                WHERE af.assoc_count IS NULL OR af.assoc_count = 0
                ORDER BY sf.file_path
            ''')
            
            models_needing_associations = cursor.fetchall()
            self.print_progress(f"Found {len(models_needing_associations)} models without associations")
            
            if self.verbose:
                if len(models_needing_associations) > 0:
                    print(f"📋 Models needing associations:")
                    for i, (_, _, model_path, model_name) in enumerate(models_needing_associations[:5]):  # Show first 5
                        print(f"   {i+1}. {model_name}")
                    if len(models_needing_associations) > 5:
                        print(f"   ... and {len(models_needing_associations) - 5} more")
                    print("")
            
            # Check for orphaned associated files (files that could belong to models but aren't linked)
            cursor.execute('''
                SELECT sf.id, sf.file_path, sf.file_name
                FROM scanned_files sf
                LEFT JOIN associated_files af ON sf.id = af.scanned_file_id
                WHERE sf.file_type IN ('text', 'image', 'video', 'json', 'unknown')
                AND af.scanned_file_id IS NULL
                ORDER BY sf.file_path
            ''')
            
            orphaned_files = cursor.fetchall()
            self.print_progress(f"Found {len(orphaned_files)} potentially orphaned associated files")
            
            if self.verbose and len(orphaned_files) > 0:
                print(f"📋 Orphaned files found:")
                for i, (_, file_path, file_name) in enumerate(orphaned_files[:5]):  # Show first 5
                    print(f"   {i+1}. {file_name}")
                if len(orphaned_files) > 5:
                    print(f"   ... and {len(orphaned_files) - 5} more")
                print("")
            
            associations_created = 0
            missing_associations_found = 0
            
            # Initialize unmatched files logging
            from datetime import datetime
            unmatched_log = []
            models_processed_count = 0
            log_filename = f"unmatched_files_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            log_path = os.path.join("processing_report", log_filename)
            
            # Ensure directory exists
            os.makedirs("processing_report", exist_ok=True)
            
            def write_unmatched_log_batch():
                """Write current unmatched log to file"""
                if unmatched_log:
                    import json
                    log_data = {
                        'timestamp': datetime.now().isoformat(),
                        'command': 'rescan-linked',
                        'batch_info': {
                            'models_processed_so_far': models_processed_count,
                            'total_models': len(models_needing_associations),
                            'unmatched_in_this_batch': len(unmatched_log)
                        },
                        'unmatched_files': unmatched_log
                    }
                    
                    # Append to existing file or create new
                    try:
                        if os.path.exists(log_path):
                            with open(log_path, 'r', encoding='utf-8') as f:
                                existing_data = json.load(f)
                            if 'all_unmatched_files' not in existing_data:
                                existing_data['all_unmatched_files'] = []
                            existing_data['all_unmatched_files'].extend(unmatched_log)
                            existing_data['last_update'] = datetime.now().isoformat()
                            existing_data['total_models_processed'] = models_processed_count
                        else:
                            existing_data = {
                                'started': datetime.now().isoformat(),
                                'command': 'rescan-linked',
                                'total_models_to_process': len(models_needing_associations),
                                'all_unmatched_files': unmatched_log,
                                'last_update': datetime.now().isoformat(),
                                'total_models_processed': models_processed_count
                            }
                        
                        with open(log_path, 'w', encoding='utf-8') as f:
                            json.dump(existing_data, f, indent=2, ensure_ascii=False)
                        
                        if self.verbose:
                            print(f"📋 Logged {len(unmatched_log)} unmatched files to {log_path}")
                        
                    except Exception as e:
                        print(f"⚠️ Error writing unmatched log: {e}")
                    
                    # Clear the batch
                    unmatched_log.clear()
            
            # Process models needing associations
            for model_file_id, model_sf_id, model_path, model_name in models_needing_associations:
                if self.verbose:
                    print(f"🔍 Processing model: {model_name}")
                    print(f"   Path: {model_path}")
                
                model_dir = os.path.dirname(model_path)
                model_base_name = os.path.splitext(model_name)[0]
                
                # Look for associated files in the same directory
                cursor.execute('''
                    SELECT sf.id, sf.file_path, sf.file_name
                    FROM scanned_files sf
                    WHERE sf.file_path LIKE ? AND sf.file_type IN ('text', 'image', 'video', 'json', 'unknown')
                ''', (f"{model_dir}%",))
                
                potential_files = cursor.fetchall()
                same_dir_files = [f for f in potential_files if os.path.dirname(f[1]) == model_dir]
                
                if self.verbose:
                    print(f"   Found {len(same_dir_files)} potential associated files in directory")
                
                # Use FileScanner's sophisticated correlation logic
                candidate_files = [assoc_path for _, assoc_path, _ in same_dir_files]
                
                # Get all associated files that should be linked to this model using server database
                try:
                    associated_files = temp_scanner.find_associated_files(model_path)
                except Exception as e:
                    if self.verbose:
                        print(f"   ⚠️ Could not run advanced correlation (file access issue): {e}")
                    associated_files = []
                
                files_linked = 0
                for assoc_sf_id, assoc_path, assoc_name in same_dir_files:
                    assoc_base_name = os.path.splitext(assoc_name)[0]
                    
                    # Check if this file should be associated using sophisticated logic
                    should_link = False
                    match_reason = ""
                    
                    # METHOD 1: Check if FileScanner correlation identified this file
                    if assoc_path in associated_files:
                        should_link = True
                        match_reason = "Advanced correlation"
                        if self.verbose:
                            print(f"   ✓ Advanced correlation match: {assoc_name}")
                    
                    # METHOD 2: Adaptive filename matching - version-aware or base name
                    elif not should_link:
                        # Smart adaptive matching: version-aware if versions exist, base name if not
                        def has_version_info(filename):
                            """Check if filename contains version information"""
                            import re
                            base = os.path.splitext(filename)[0]
                            if base.endswith('.metadata'):
                                base = base[:-9]
                            
                            # Look for various version patterns:
                            # - v1.0, v2.1 (standard semantic versioning)
                            # - V11, V23 (single version numbers)
                            # - V11_0.85_V8_0.15 (complex version combinations)
                            version_patterns = [
                                r'v\d+\.\d+',                    # v1.0, v2.1
                                r'V\d+(?:_[\d\.]+)*(?:_V\d+)*',  # V11, V11_0.85_V8_0.15
                                r'_v\d+',                        # _v1, _v2
                            ]
                            
                            for pattern in version_patterns:
                                if re.search(pattern, base, re.IGNORECASE):
                                    return True
                            return False
                        
                        def extract_adaptive_name(filename, use_version_aware=True):
                            """Extract filename - version-aware or base name depending on context"""
                            base = os.path.splitext(filename)[0]
                            # Remove .metadata suffix
                            if base.endswith('.metadata'):
                                base = base[:-9]
                            
                            if use_version_aware:
                                # Keep version info
                                return base
                            else:
                                # Remove version and hash info for base name matching
                                import re
                                cleaned = re.sub(r'[_\s]*v?\d+\.\d+[_\s]*\d*', '', base)
                                cleaned = re.sub(r'[_\s]*\d{7,}[_\s]*', '', cleaned)
                                cleaned = re.sub(r'[_\s]+$', '', cleaned).strip()
                                return cleaned
                        
                        # Check if any files in this directory have version info
                        dir_files = [assoc_name for _, _, assoc_name in same_dir_files] + [model_name]
                        has_versions = any(has_version_info(f) for f in dir_files)
                        
                        # Pre-calculate model adaptive name and match type for both local and target directory processing
                        model_adaptive = extract_adaptive_name(model_name, use_version_aware=has_versions)
                        match_type = "Version-aware" if has_versions else "Base name"
                        
                        if has_versions:
                            # Use version-aware matching
                            model_match_name = extract_adaptive_name(model_name, use_version_aware=True)
                            assoc_match_name = extract_adaptive_name(assoc_name, use_version_aware=True)
                        else:
                            # Use base name matching (no versions in directory)
                            model_match_name = extract_adaptive_name(model_name, use_version_aware=False)
                            assoc_match_name = extract_adaptive_name(assoc_name, use_version_aware=False)
                        
                        # Check if adaptive filenames match for media/metadata files
                        if (model_match_name.lower() == assoc_match_name.lower() and 
                            assoc_name.lower().endswith(('.json', '.txt', '.metadata.json', '.mp4', '.webp', '.png', '.jpg', '.jpeg'))):
                            should_link = True
                            match_reason = f"{match_type} filename match"
                            if self.verbose:
                                print(f"   ✓ {match_type} filename match: {assoc_name}")
                                print(f"      Matched: '{model_match_name}' == '{assoc_match_name}' (versions_detected: {has_versions})")
                    
                    # METHOD 3: Basic exact name match (fallback)
                    elif assoc_base_name == model_base_name:
                        should_link = True
                        match_reason = "Exact name match"
                        if self.verbose:
                            print(f"   ✓ Exact name match: {assoc_name}")
                    
                    # METHOD 4: Single model scenario (fallback)
                    elif len(models_needing_associations) == 1:  # Only one model in this batch
                        # Check if only one model in this directory
                        cursor.execute('''
                            SELECT COUNT(*) FROM model_files mf
                            JOIN scanned_files sf ON mf.scanned_file_id = sf.id
                            WHERE sf.file_path LIKE ?
                        ''', (f"{model_dir}%",))
                        models_in_dir = cursor.fetchone()[0]
                        if models_in_dir == 1:
                            should_link = True
                            match_reason = "Single model scenario"
                            if self.verbose:
                                print(f"   ✓ Single model scenario: {assoc_name}")
                    
                    if should_link:
                        # Check if association already exists
                        cursor.execute('''
                            SELECT 1 FROM associated_files 
                            WHERE model_file_id = ? AND scanned_file_id = ?
                        ''', (model_file_id, assoc_sf_id))
                        
                        if not cursor.fetchone():
                            assoc_type = self._determine_association_type(assoc_path)
                            cursor.execute('''
                                INSERT INTO associated_files 
                                (model_file_id, scanned_file_id, association_type, source_path, is_moved)
                                VALUES (?, ?, ?, ?, 0)
                            ''', (model_file_id, assoc_sf_id, assoc_type, assoc_path))
                            associations_created += 1
                            files_linked += 1
                            if self.verbose:
                                print(f"   🔗 Linked {assoc_type}: {assoc_name}")
                        else:
                            if self.verbose:
                                print(f"   ⚠️ Already linked: {assoc_name}")
                        
                        missing_associations_found += 1
                    else:
                        # Log unmatched file with detailed information
                        unmatched_entry = {
                            'model_name': model_name,
                            'model_path': model_path,
                            'model_base_name': model_base_name,
                            'unmatched_file': assoc_name,
                            'unmatched_path': assoc_path,
                            'unmatched_base_name': assoc_base_name,
                            'attempted_methods': [],
                            'reason': 'No correlation methods succeeded'
                        }
                        
                        # Document what was attempted
                        if assoc_path in associated_files:
                            unmatched_entry['attempted_methods'].append('Advanced correlation - FAILED')
                        else:
                            unmatched_entry['attempted_methods'].append('Advanced correlation - NOT FOUND')
                        
                        # Check adaptive filename matching attempt
                        def has_version_info_log(filename):
                            import re
                            base = os.path.splitext(filename)[0]
                            if base.endswith('.metadata'):
                                base = base[:-9]
                            
                            # Look for various version patterns (same as main function)
                            version_patterns = [
                                r'v\d+\.\d+',                    # v1.0, v2.1
                                r'V\d+(?:_[\d\.]+)*(?:_V\d+)*',  # V11, V11_0.85_V8_0.15
                                r'_v\d+',                        # _v1, _v2
                            ]
                            
                            for pattern in version_patterns:
                                if re.search(pattern, base, re.IGNORECASE):
                                    return True
                            return False
                        
                        def extract_adaptive_name_log(filename, use_version_aware=True):
                            base = os.path.splitext(filename)[0]
                            if base.endswith('.metadata'):
                                base = base[:-9]
                            
                            if use_version_aware:
                                return base
                            else:
                                import re
                                cleaned = re.sub(r'[_\s]*v?\d+\.\d+[_\s]*\d*', '', base)
                                cleaned = re.sub(r'[_\s]*\d{7,}[_\s]*', '', cleaned)
                                cleaned = re.sub(r'[_\s]+$', '', cleaned).strip()
                                return cleaned
                        
                        # Check if directory has versions
                        dir_files = [f[2] for f in same_dir_files] + [model_name]
                        has_versions = any(has_version_info_log(f) for f in dir_files)
                        
                        model_adaptive = extract_adaptive_name_log(model_name, use_version_aware=has_versions)
                        assoc_adaptive = extract_adaptive_name_log(assoc_name, use_version_aware=has_versions)
                        match_type = "Version-aware" if has_versions else "Base name"
                        
                        if (model_adaptive.lower() == assoc_adaptive.lower() and 
                            assoc_name.lower().endswith(('.json', '.txt', '.metadata.json', '.mp4', '.webp', '.png', '.jpg', '.jpeg'))):
                            unmatched_entry['attempted_methods'].append(f'{match_type} filename match - MATCHED ("{model_adaptive}" == "{assoc_adaptive}") but failed other checks')
                        else:
                            unmatched_entry['attempted_methods'].append(f'{match_type} filename match - FAILED ("{model_adaptive}" != "{assoc_adaptive}")')
                        
                        if assoc_base_name == model_base_name:
                            unmatched_entry['attempted_methods'].append('Exact name match - MATCHED but failed other checks')
                        else:
                            unmatched_entry['attempted_methods'].append(f'Exact name match - FAILED ("{assoc_base_name}" != "{model_base_name}")')
                        
                        unmatched_entry['attempted_methods'].append(f'Single model scenario - {"PASSED" if len(models_needing_associations) == 1 else "FAILED (multiple models)"}')
                        
                        unmatched_log.append(unmatched_entry)
                        
                        if self.verbose:
                            print(f"   ❌ No match: {assoc_name}")
                            print(f"      Attempted: {', '.join(unmatched_entry['attempted_methods'])}")
                
                # STEP 2: Check target directory for matching files
                target_dir = self.get_destination_directory()
                target_files_linked = 0
                
                if target_dir and os.path.exists(target_dir):
                    if self.verbose:
                        print(f"   🎯 Checking target directory: {os.path.basename(target_dir)}")
                    
                    # METHOD 1: Quick database check first
                    target_matches_from_db = []
                    try:
                        cursor.execute('''
                            SELECT sf.id, sf.file_path, sf.file_name, sf.blake3_hash, sf.metadata_json
                            FROM scanned_files sf
                            WHERE sf.file_path LIKE ? 
                            AND sf.file_name NOT LIKE '%.safetensors'
                            AND sf.file_name NOT LIKE '%.ckpt'
                            AND sf.file_name NOT LIKE '%.pt'
                            AND sf.file_name NOT LIKE '%.pth'
                            AND (sf.file_name LIKE '%.json' OR sf.file_name LIKE '%.txt' OR 
                                 sf.file_name LIKE '%.webp' OR sf.file_name LIKE '%.png' OR
                                 sf.file_name LIKE '%.jpg' OR sf.file_name LIKE '%.jpeg' OR
                                 sf.file_name LIKE '%.mp4')
                        ''', (f"{target_dir}%",))
                        target_matches_from_db = cursor.fetchall()
                        
                        if self.verbose:
                            print(f"      Found {len(target_matches_from_db)} potential files in database")
                    except Exception as e:
                        if self.verbose:
                            print(f"      ⚠️ Database query failed: {e}")
                    
                    # Check database matches using adaptive filename matching
                    for target_sf_id, target_path, target_name, target_hash, target_metadata in target_matches_from_db:
                        target_base_name = os.path.splitext(target_name)[0]
                        if target_base_name.endswith('.metadata'):
                            target_base_name = target_base_name[:-9]
                        
                        should_link_target = False
                        match_reason_target = ""
                        
                        # Use same adaptive matching logic as local files
                        target_adaptive = extract_adaptive_name(target_name, use_version_aware=has_versions)
                        
                        if model_adaptive.lower() == target_adaptive.lower():
                            should_link_target = True
                            match_reason_target = f"Target directory {match_type} filename match"
                            if self.verbose:
                                print(f"      ✓ Target {match_type} match: {target_name}")
                                print(f"         Matched: '{model_adaptive}' == '{target_adaptive}'")
                        
                        # Verify with metadata if available
                        if should_link_target and target_metadata:
                            try:
                                metadata = json.loads(target_metadata)
                                # Enhanced metadata verification
                                if ('model_name' in metadata and metadata['model_name'] and
                                    model_base_name.lower() not in metadata['model_name'].lower()):
                                    should_link_target = False
                                    match_reason_target += " (metadata name mismatch)"
                                    if self.verbose:
                                        print(f"         ⚠️ Metadata name mismatch: model='{model_base_name}' vs metadata='{metadata.get('model_name', 'N/A')}'")
                                elif self.verbose:
                                    print(f"         ✓ Metadata verification passed")
                            except (json.JSONDecodeError, KeyError) as e:
                                if self.verbose:
                                    print(f"         ⚠️ Metadata parse error: {e}")
                        
                        if should_link_target:
                            # Check if association already exists
                            cursor.execute('''
                                SELECT 1 FROM associated_files 
                                WHERE model_file_id = ? AND scanned_file_id = ?
                            ''', (model_file_id, target_sf_id))
                            
                            if not cursor.fetchone():
                                assoc_type = self._determine_association_type(target_path)
                                cursor.execute('''
                                    INSERT INTO associated_files 
                                    (model_file_id, scanned_file_id, association_type, source_path, is_moved)
                                    VALUES (?, ?, ?, ?, 0)
                                ''', (model_file_id, target_sf_id, assoc_type, target_path))
                                associations_created += 1
                                target_files_linked += 1
                                files_linked += 1
                                if self.verbose:
                                    print(f"      🔗 Linked target {assoc_type}: {target_name}")
                            else:
                                if self.verbose:
                                    print(f"      ⚠️ Already linked: {target_name}")
                    
                    # METHOD 2: Physical file scan if no database matches or need fallback
                    if len(target_matches_from_db) == 0:
                        if self.verbose:
                            print(f"      � Scanning target directory physically...")
                        
                        try:
                            for root, dirs, files in os.walk(target_dir):
                                for file in files:
                                    if file.lower().endswith(('.json', '.txt', '.webp', '.png', '.jpg', '.jpeg', '.mp4')):
                                        file_path = os.path.join(root, file)
                                        file_base = os.path.splitext(file)[0]
                                        if file_base.endswith('.metadata'):
                                            file_base = file_base[:-9]
                                        
                                        # Use adaptive matching
                                        file_adaptive = extract_adaptive_name(file, use_version_aware=has_versions)
                                        
                                        if model_adaptive.lower() == file_adaptive.lower():
                                            # Verify with metadata if it's a JSON file
                                            metadata_verified = True
                                            if file.lower().endswith('.json'):
                                                try:
                                                    with open(file_path, 'r', encoding='utf-8') as f:
                                                        metadata = json.load(f)
                                                    if ('model_name' in metadata and metadata['model_name'] and
                                                        model_base_name.lower() not in metadata['model_name'].lower()):
                                                        metadata_verified = False
                                                        if self.verbose:
                                                            print(f"         ⚠️ Physical file metadata mismatch: {file}")
                                                except Exception as e:
                                                    if self.verbose:
                                                        print(f"         ⚠️ Could not read metadata from {file}: {e}")
                                            
                                            if metadata_verified:
                                                # Check if this file is already in database and linked
                                                cursor.execute('''
                                                    SELECT sf.id FROM scanned_files sf
                                                    WHERE sf.file_path = ?
                                                ''', (file_path,))
                                                existing_sf = cursor.fetchone()
                                                
                                                if existing_sf:
                                                    # Check if already linked
                                                    cursor.execute('''
                                                        SELECT 1 FROM associated_files 
                                                        WHERE model_file_id = ? AND scanned_file_id = ?
                                                    ''', (model_file_id, existing_sf[0]))
                                                    
                                                    if not cursor.fetchone():
                                                        assoc_type = self._determine_association_type(file_path)
                                                        cursor.execute('''
                                                            INSERT INTO associated_files 
                                                            (model_file_id, scanned_file_id, association_type, source_path, is_moved)
                                                            VALUES (?, ?, ?, ?, 0)
                                                        ''', (model_file_id, existing_sf[0], assoc_type, file_path))
                                                        associations_created += 1
                                                        target_files_linked += 1
                                                        files_linked += 1
                                                        if self.verbose:
                                                            print(f"      🔗 Linked target (physical): {file}")
                                                else:
                                                    if self.verbose:
                                                        print(f"      📝 Found unscanned target file: {file}")
                                        
                                        # Limit physical scan to prevent excessive processing
                                        if target_files_linked >= 10:  # Reasonable limit
                                            break
                                if target_files_linked >= 10:
                                    break
                        except Exception as e:
                            if self.verbose:
                                print(f"      ⚠️ Physical scan error: {e}")
                    
                    if self.verbose and target_files_linked > 0:
                        print(f"      🎯 Target directory result: {target_files_linked} associations created")
                elif self.verbose:
                    if not target_dir:
                        print(f"   ⚠️ No destination directory configured")
                    else:
                        print(f"   ⚠️ Target directory not found: {target_dir}")
                
                if self.verbose:
                    total_message = f"📊 Result: {files_linked} total associations created for {model_name}"
                    if target_files_linked > 0:
                        total_message += f" ({files_linked - target_files_linked} local + {target_files_linked} target)"
                    print(f"   {total_message}")
                    print("")
                
                # Increment model counter and write log every 5 models
                models_processed_count += 1
                if models_processed_count % 5 == 0:
                    write_unmatched_log_batch()
                    if self.verbose:
                        print(f"📋 Processed {models_processed_count}/{len(models_needing_associations)} models, log updated")
            
            # Write any remaining unmatched files
            if unmatched_log:
                write_unmatched_log_batch()
            
            # Verify existing associations are still valid
            cursor.execute('''
                SELECT af.id, af.model_file_id, af.scanned_file_id, af.source_path
                FROM associated_files af
                JOIN model_files mf ON af.model_file_id = mf.id
                JOIN scanned_files sf ON af.scanned_file_id = sf.id
            ''')
            
            existing_associations = cursor.fetchall()
            
            if self.verbose:
                print(f"🔍 Verifying {len(existing_associations)} existing associations...")
            
            invalid_associations = 0
            
            for assoc_id, model_file_id, scanned_file_id, source_path in existing_associations:
                # Check if the file still exists
                if not os.path.exists(source_path):
                    cursor.execute('DELETE FROM associated_files WHERE id = ?', (assoc_id,))
                    invalid_associations += 1
                    if self.verbose:
                        print(f"   ❌ Removed invalid association: {os.path.basename(source_path)}")
                        print(f"      (File no longer exists at: {source_path})")
                elif self.verbose:
                    print(f"   ✓ Valid association: {os.path.basename(source_path)}")
            
            if self.verbose and invalid_associations > 0:
                print(f"📊 Removed {invalid_associations} invalid associations")
            elif self.verbose:
                print(f"📊 All existing associations are valid")
            
            conn.commit()
            
            self.print_progress(f"Verification complete:")
            self.print_progress(f"  - Models checked: {len(models_needing_associations):,}")
            self.print_progress(f"  - Missing associations found: {missing_associations_found:,}")
            self.print_progress(f"  - New associations created: {associations_created:,}")
            self.print_progress(f"  - Invalid associations removed: {invalid_associations:,}")
            self.print_progress(f"  - Orphaned files detected: {len(orphaned_files):,}")
            
            if associations_created > 0:
                self.print_progress(f"✓ Fixed {associations_created} missing associations")
            if invalid_associations > 0:
                self.print_progress(f"✓ Cleaned up {invalid_associations} invalid associations")
            if associations_created == 0 and invalid_associations == 0:
                self.print_progress("✓ All associations are correct, no changes needed")
            
            # Final log summary
            if os.path.exists(log_path):
                try:
                    import json
                    with open(log_path, 'r', encoding='utf-8') as f:
                        final_log_data = json.load(f)
                    total_unmatched = len(final_log_data.get('all_unmatched_files', []))
                    self.print_progress(f"📋 Final unmatched files log: {log_path}")
                    self.print_progress(f"   Total unmatched files: {total_unmatched}")
                    if self.verbose:
                        print(f"\n📋 Complete Unmatched Files Analysis:")
                        print(f"   Log file: {log_path}")
                        print(f"   Total models processed: {models_processed_count}")
                        print(f"   Total unmatched files: {total_unmatched}")
                        print(f"   Review the JSON file for detailed analysis")
                except Exception as e:
                    self.print_progress(f"⚠️ Error reading final log summary: {e}")
            
            return True
            
        except Exception as e:
            print(f"ERROR in verify linked files: {e}")
            self.stats['errors'] += 1
            return False
    
    def step_rebuild_all_links(self) -> bool:
        """Step: Completely rebuild all file associations from scratch"""
        try:
            self.print_step("REBUILD ALL LINKS", 
                           "Completely rebuilding all file associations from scratch")
            
            # Initialize scanner to access database
            if not hasattr(self, 'scanner'):
                from file_scanner import FileScanner
                self.scanner = FileScanner(self.config_path)
            
            conn = self.scanner.db.conn
            cursor = conn.cursor()
            
            # Get all model files from scanned_files (not just those with extracted metadata)
            cursor.execute('''
                SELECT sf.id, sf.file_name, sf.file_path
                FROM scanned_files sf
                WHERE sf.file_type = 'model' AND sf.extension IN ('.safetensors', '.ckpt', '.pt', '.pth')
                ORDER BY sf.file_path
            ''')
            
            model_files = cursor.fetchall()
            self.print_progress(f"Found {len(model_files)} model files to process")
            
            # Get all non-model files that could be associated
            cursor.execute('''
                SELECT sf.id, sf.file_path, sf.file_name
                FROM scanned_files sf
                WHERE sf.file_type IN ('text', 'image', 'video', 'json', 'unknown')
                ORDER BY sf.file_path
            ''')
            
            potential_associated_files = cursor.fetchall()
            self.print_progress(f"Found {len(potential_associated_files)} potential associated files")
            
            # Clear existing associations to rebuild them
            cursor.execute('DELETE FROM associated_files')
            self.print_progress("Cleared all existing file associations")
            
            # Clear temporary associations too
            cursor.execute('DROP TABLE IF EXISTS temp_file_associations')
            cursor.execute('DROP TABLE IF EXISTS temp_associations')
            
            # Create a table to track model->associated file relationships
            cursor.execute('''
                CREATE TEMPORARY TABLE temp_associations (
                    model_scanned_file_id INTEGER,
                    assoc_scanned_file_id INTEGER,
                    association_type TEXT,
                    source_path TEXT
                )
            ''')
            
            linked_count = 0
            processed_count = 0
            
            # Process each model
            for model_sf_id, model_file_name, model_path in model_files:
                processed_count += 1
                if processed_count % 1000 == 0:
                    self.print_progress(f"Processed {processed_count:,}/{len(model_files):,} models...")
                
                # Extract directory and base name from model path
                model_dir = os.path.dirname(model_path)
                model_base_name = os.path.splitext(model_file_name)[0]
                
                # Find associated files for this model
                associated_files = []
                
                # 1. Same directory, exact name match
                for sf_id, sf_path, sf_name in potential_associated_files:
                    sf_dir = os.path.dirname(sf_path)
                    if sf_dir != model_dir:
                        continue
                    
                    sf_base_name = os.path.splitext(sf_name)[0]
                    
                    # Exact name match
                    if sf_base_name == model_base_name:
                        assoc_type = self._determine_association_type(sf_path)
                        associated_files.append((sf_id, sf_path, assoc_type))
                
                # 2. If only one model in directory, link all files in that directory
                same_dir_models = [m for m in model_files if os.path.dirname(m[2]) == model_dir]
                if len(same_dir_models) == 1:
                    for sf_id, sf_path, sf_name in potential_associated_files:
                        sf_dir = os.path.dirname(sf_path)
                        if sf_dir == model_dir:
                            sf_base_name = os.path.splitext(sf_name)[0]
                            # Avoid duplicates from exact matches above
                            if sf_base_name != model_base_name:
                                assoc_type = self._determine_association_type(sf_path)
                                associated_files.append((sf_id, sf_path, assoc_type))
                
                # Insert associations into temporary table
                for sf_id, sf_path, assoc_type in associated_files:
                    cursor.execute('''
                        INSERT INTO temp_associations 
                        (model_scanned_file_id, assoc_scanned_file_id, association_type, source_path)
                        VALUES (?, ?, ?, ?)
                    ''', (model_sf_id, sf_id, assoc_type, sf_path))
                    linked_count += 1
            
            # Now insert into associated_files table for models that exist in model_files
            cursor.execute('''
                INSERT INTO associated_files 
                (model_file_id, scanned_file_id, association_type, source_path, is_moved)
                SELECT mf.id, ta.assoc_scanned_file_id, ta.association_type, ta.source_path, 0
                FROM temp_associations ta
                JOIN model_files mf ON mf.scanned_file_id = ta.model_scanned_file_id
            ''')
            
            associated_with_metadata = cursor.rowcount
            
            conn.commit()
            
            self.print_progress(f"Complete rebuild finished:")
            self.print_progress(f"  - Models processed: {processed_count:,}")
            self.print_progress(f"  - Total associations found: {linked_count:,}")
            self.print_progress(f"  - Associations linked to extracted metadata: {associated_with_metadata:,}")
            if processed_count > 0:
                self.print_progress(f"  - Average associations per model: {linked_count/processed_count:.1f}")
            
            if associated_with_metadata < linked_count:
                self.print_progress(f"  - Note: {linked_count - associated_with_metadata:,} associations are pending metadata extraction")
            
            return True
            
        except Exception as e:
            print(f"ERROR in rebuild all links: {e}")
            self.stats['errors'] += 1
            return False

    def step_rescan_and_repair_metadata_text(self) -> bool:
        """Step: Rescan and repair metadata/text file paths with intelligent path correction"""
        try:
            self.print_step("RESCAN AND REPAIR METADATA/TEXT FILE PATHS", 
                           "Intelligently scan filesystem and repair database path mismatches")
            
            # Initialize scanner to access database
            if not hasattr(self, 'scanner'):
                from file_scanner import FileScanner
                self.scanner = FileScanner(self.config_path)
            
            results = self.scanner.rescan_and_repair_text_file_paths()
            
            # Display results
            self.print_progress(f"Files scanned: {results.get('files_scanned', 0):,}")
            self.print_progress(f"Paths checked: {results.get('paths_checked', 0):,}")
            self.print_progress(f"Paths corrected: {results.get('paths_corrected', 0):,}")
            self.print_progress(f"Database updates: {results.get('database_updates', 0):,}")
            
            if results.get('errors', 0) > 0:
                print(f"⚠️  Errors encountered: {results.get('errors', 0)}")
                self.stats['errors'] += results.get('errors', 0)
            
            return True
            
        except Exception as e:
            print(f"ERROR in rescan and repair metadata/text: {e}")
            self.stats['errors'] += 1
            return False

    def step_migrate_tables(self) -> bool:
        """Migrate model files from scanned_files to model_files table"""
        try:
            self.print_step("MIGRATE TABLES", 
                          "Migrating model files from scanned_files to model_files table")
            
            import sqlite3
            
            # Connect to the server database using config
            server_db_path = self.get_server_database_path()
            if not os.path.exists(server_db_path):
                print(f"❌ Server database not found: {server_db_path}")
                return False
            
            conn = sqlite3.connect(server_db_path, timeout=30.0)
            cursor = conn.cursor()
            
            # Get all model files from scanned_files that don't have entries in model_files
            self.print_progress("Finding model files to migrate...")
            cursor.execute('''
                SELECT sf.id, sf.file_path, sf.file_name, sf.file_size, sf.sha256, sf.blake3
                FROM scanned_files sf
                LEFT JOIN model_files mf ON sf.id = mf.scanned_file_id
                WHERE sf.file_type = 'model' AND mf.id IS NULL
                ORDER BY sf.file_path
            ''')
            
            models_to_migrate = cursor.fetchall()
            total_models = len(models_to_migrate)
            
            if total_models == 0:
                self.print_progress("✅ No models need migration - all model files already in model_files table")
                conn.close()
                return True
            
            self.print_progress(f"Found {total_models} model files to migrate")
            
            # Migrate in batches for better performance
            batch_size = 1000
            migrated_count = 0
            
            for i in range(0, total_models, batch_size):
                batch = models_to_migrate[i:i + batch_size]
                batch_end = min(i + batch_size, total_models)
                
                self.print_progress(f"Migrating batch {i//batch_size + 1}: models {i+1}-{batch_end} of {total_models}")
                
                # Prepare batch insert data
                insert_data = []
                for scanned_file_id, file_path, file_name, file_size, sha256, blake3 in batch:
                    # Extract basic model info from filename
                    model_name = os.path.splitext(file_name)[0]
                    
                    # Determine basic model type from extension
                    if file_path.lower().endswith('.safetensors'):
                        if 'checkpoint' in file_path.lower() or 'model' in file_path.lower():
                            model_type = 'CHECKPOINT'
                        else:
                            model_type = 'LORA'  # Default for .safetensors
                    elif file_path.lower().endswith('.pt'):
                        model_type = 'LORA'
                    elif file_path.lower().endswith('.ckpt'):
                        model_type = 'CHECKPOINT'
                    else:
                        model_type = 'LORA'  # Default
                    
                    insert_data.append((
                        scanned_file_id,
                        model_name,
                        'Unknown',  # base_model - will be enhanced later
                        model_type,
                        None,  # civitai_id
                        None,  # version_id
                        file_path,  # source_path
                        None,  # target_path
                        0,  # is_duplicate
                        None,  # duplicate_group_id
                        None,  # metadata_json
                        0,  # has_civitai_info
                        0,  # has_metadata_json
                        'pending',  # status
                        None  # error_message
                    ))
                
                # Batch insert into model_files
                cursor.executemany('''
                    INSERT INTO model_files 
                    (scanned_file_id, model_name, base_model, model_type, civitai_id, version_id, 
                     source_path, target_path, is_duplicate, duplicate_group_id, metadata_json, 
                     has_civitai_info, has_metadata_json, status, error_message, 
                     created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 
                            strftime('%s', 'now'), strftime('%s', 'now'))
                ''', insert_data)
                
                migrated_count += len(batch)
                
                # Commit this batch
                conn.commit()
                
                if migrated_count % 5000 == 0:
                    self.print_progress(f"  ✅ Migrated {migrated_count} models so far...")
            
            conn.close()
            
            self.print_progress(f"✅ Migration complete!")
            self.print_progress(f"Migrated {migrated_count} model files from scanned_files to model_files")
            self.print_progress(f"Note: Models migrated with basic info - run --step metadata to enhance with full metadata")
            
            self.stats['models_migrated'] = migrated_count
            
            return True
            
        except Exception as e:
            print(f"ERROR in table migration: {e}")
            import traceback
            traceback.print_exc()
            self.stats['errors'] += 1
            return False
    
    def _determine_association_type(self, file_path: str) -> str:
        """Determine the type of association based on file extension and name"""
        file_name = os.path.basename(file_path).lower()
        
        if file_name.endswith('.civitai.info'):
            return 'civitai_info'
        elif file_name.endswith('.metadata.json'):
            return 'metadata'
        elif any(file_name.endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.webp', '.gif']):
            return 'image'
        elif any(file_name.endswith(ext) for ext in ['.mp4', '.avi', '.mov', '.mkv']):
            return 'video'
        elif file_name.endswith('.txt'):
            return 'text'
        else:
            return 'other'
    
    def generate_final_report(self):
        """Generate final processing report"""
        end_time = time.time()
        duration = end_time - self.start_time
        
        print(f"\n{'='*60}")
        print("PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Total processing time: {duration:.2f} seconds")
        print(f"Mode: {'DRY RUN' if self.dry_run else 'LIVE'}")
        
        print(f"\nSummary:")
        print(f"  Files scanned: {self.stats['files_scanned']}")
        print(f"  Model files found: {self.stats['models_found']}")
        print(f"  Duplicate files found: {self.stats['duplicates_found']}")
        print(f"  Models sorted: {self.stats['models_sorted']}")
        print(f"  Civitai.info files generated: {self.stats['civitai_files_generated']}")
        print(f"  Errors encountered: {self.stats['errors']}")
        
        # Stop console capture and get output
        captured_output = self.console_capture.stop_capture()
        
        # Save report to file
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': duration,
            'dry_run': self.dry_run,
            'config_file': self.config_path,
            'stats': self.stats,
            'console_output': captured_output
        }
        
        # Create processing_report directory if it doesn't exist
        report_dir = "processing_report"
        os.makedirs(report_dir, exist_ok=True)
        
        report_file = f"processing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path = os.path.join(report_dir, report_file)
        try:
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2)
            print(f"\nDetailed report saved to: {report_path}")
        except Exception as e:
            print(f"Warning: Could not save report file: {e}")
    
    def run_full_workflow(self) -> bool:
        """Run the complete model sorting workflow"""
        try:
            # Start capturing console output
            self.console_capture.start_capture()
            
            self.print_progress("Starting Model Sorter workflow...")
            self.print_progress(f"Configuration: {self.config_path}")
            self.print_progress(f"Mode: {'DRY RUN' if self.dry_run else 'LIVE PROCESSING'}")
            
            # Step 1: Scan files
            if not self.step_1_scan_files():
                print("FAILED: File scanning failed")
                return False
            
            # Step 2: Extract metadata
            if not self.step_2_extract_metadata():
                print("FAILED: Metadata extraction failed")
                return False
            
            # Step 3: Detect duplicates
            if not self.step_3_detect_duplicates():
                print("FAILED: Duplicate detection failed")
                return False
            
            # Step 4: Sort models
            if not self.step_4_sort_models():
                print("FAILED: Model sorting failed")
                return False
            
            # Step 5: Generate civitai.info files
            if not self.step_5_generate_civitai_files():
                print("FAILED: Civitai.info generation failed")
                return False
            
            # Generate final report
            self.generate_final_report()
            
            return True
            
        except KeyboardInterrupt:
            print("\nProcessing interrupted by user")
            return False
        except Exception as e:
            print(f"CRITICAL ERROR: {e}")
            self.stats['errors'] += 1
            return False
    
    def run_individual_step(self, step_name: str) -> bool:
        """Run an individual step of the workflow"""
        steps = {
            'scan': self.step_1_scan_files,
            'metadata': self.step_2_extract_metadata,
            'duplicates': self.step_3_detect_duplicates,
            'sort': self.step_4_sort_models,
            'civitai': self.step_5_generate_civitai_files,
            'rescan': self.step_rescan_metadata,
            'extract-metadata': self.step_extract_civitai_metadata,
            'retry-failed': self.step_retry_failed_metadata,
            'migrate-paths': None,  # Special handling needed
            'migrate-tables': self.step_migrate_tables,
            'recover-missing': self.step_recover_missing_files,
            'force-blake3': self.step_force_blake3_rescan,
            'update-tables': self.step_update_tables,
            'rescan-linked': self.step_rescan_linked_files,
            'rescan-linked-rebuild': self.step_rebuild_all_links,
            'rescan-and-repair-metadata-text': self.step_rescan_and_repair_metadata_text,
            'rebuild-links': self.step_rebuild_all_links  # deprecated alias
        }
        
        if step_name not in steps:
            print(f"Unknown step: {step_name}")
            print(f"Available steps: {', '.join(steps.keys())}")
            return False
        
        # Special handling for migrate-paths step which requires parameters
        if step_name == 'migrate-paths':
            # This should be called from main() with proper arguments
            raise ValueError("migrate-paths step requires special handling with arguments")
        
        return steps[step_name]()
    
    def close(self):
        """Clean up resources"""
        try:
            if hasattr(self, 'scanner'):
                self.scanner.close()
            if hasattr(self, 'metadata_extractor'):
                self.metadata_extractor.close()
            if hasattr(self, 'duplicate_detector'):
                self.duplicate_detector.close()
            if hasattr(self, 'model_sorter'):
                self.model_sorter.close()
            if hasattr(self, 'civitai_generator'):
                self.civitai_generator.close()
        except Exception as e:
            print(f"Warning during cleanup: {e}")


def main():
    """Main entry point"""
    # Handle broken pipe errors (e.g., when output is piped to head)
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    
    parser = argparse.ArgumentParser(
        description='Model Sorter - Complete workflow for organizing stable diffusion models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete workflow in dry-run mode
  python model_sorter_main.py --dry-run
  
  # Run complete workflow live
  python model_sorter_main.py
  
  # Run only specific steps
  python model_sorter_main.py --step scan
  python model_sorter_main.py --step scan --force-rescan  # Force scan all files
  python model_sorter_main.py --step metadata
  python model_sorter_main.py --step duplicates
  python model_sorter_main.py --step sort --dry-run
  python model_sorter_main.py --step civitai
  python model_sorter_main.py --step rescan
  python model_sorter_main.py --step rescan-linked           # Verify & fix associations (recommended)
  python model_sorter_main.py --step rescan-linked-rebuild   # Complete rebuild (if corrupted)
  python model_sorter_main.py --rescan-linked                # Auto-run verify & fix
  python model_sorter_main.py --rescan-linked-rebuild        # Auto-run complete rebuild
  python model_sorter_main.py --rescan-and-repair-metadata-text  # Auto-run path repair
  python model_sorter_main.py --rebuild-links                # Auto-run complete rebuild [DEPRECATED]
  python model_sorter_main.py --extract-metadata-limit 1000  # Extract from 1000 files
  python model_sorter_main.py --extract-metadata-all         # Extract from ALL files
  python model_sorter_main.py --extract-metadata-all-rescan  # Extract from ALL files (including already scanned)
  
  # Use custom config
  python model_sorter_main.py --config /path/to/config.ini

Configuration Options:
  max_file_size: Set maximum file size to scan (in bytes, 0=no limit)
    - Examples: 1073741824 (1GB), 2147483648 (2GB), 5368709120 (5GB)
    - Files exceeding this limit will be skipped during scanning
  database_commit_batch_size: Number of files to batch before database commit
    - Higher values improve performance but use more memory
    - Default: 1000 files per batch
        """)
    
    parser.add_argument('--config', '-c', default='config.ini',
                       help='Configuration file path (default: config.ini)')
    parser.add_argument('--dry-run', '-n', action='store_true',
                       help='Show what would be done without actually moving files')
    parser.add_argument('--step', '-s', 
                       choices=['scan', 'metadata', 'duplicates', 'sort', 'civitai', 'rescan', 'extract-metadata', 'retry-failed', 'migrate-paths', 'recover-missing', 'force-blake3', 'update-tables', 'rescan-linked', 'rescan-linked-rebuild', 'rescan-and-repair-metadata-text'],
                       help='Run only a specific step instead of full workflow')
    parser.add_argument('--rescan-linked', action='store_true',
                       help='Verify and correct file associations (incremental check, recommended)')
    parser.add_argument('--rescan-linked-rebuild', action='store_true',
                       help='Completely rebuild all file associations from scratch (use if database is corrupted)')
    parser.add_argument('--rebuild-links', action='store_true',
                       help='[DEPRECATED] Use --rescan-linked-rebuild instead')
    parser.add_argument('--rescan-and-repair-metadata-text', action='store_true',
                       help='Intelligently scan and repair metadata/text file path mismatches')
    parser.add_argument('--force-rescan', '-f', action='store_true',
                       help='Force rescan of all files, ignoring database cache (use with --step scan)')
    parser.add_argument('--extract-metadata-limit', type=int, 
                       help='Extract enhanced metadata from N files (automatically runs extract-metadata step)')
    parser.add_argument('--extract-metadata-all', action='store_true',
                       help='Extract enhanced metadata from ALL files (automatically runs extract-metadata step)')
    parser.add_argument('--extract-metadata-all-rescan', action='store_true',
                       help='Extract enhanced metadata from ALL files, including already scanned ones (automatically runs extract-metadata step)')
    parser.add_argument('--update-tables', action='store_true',
                       help='Check database completeness and rescan files with missing table entries')
    parser.add_argument('--skip-folders', action='store_true', default=True,
                       help='Skip directories that have already been fully scanned (default: enabled)')
    parser.add_argument('--no-skip-folders', action='store_false', dest='skip_folders',
                       help='Disable directory skipping (force scan all directories)')
    parser.add_argument('--folder-limit', type=int, metavar='N',
                       help='Limit scanning to N folders per run for incremental processing (useful for large directories)')
    parser.add_argument('--retry-failed', action='store_true',
                       help='Retry metadata extraction for files that previously failed due to missing columns')
    parser.add_argument('--migrate-paths', nargs=2, metavar=('OLD_PREFIX', 'NEW_PREFIX'),
                       help='Migrate database paths from old prefix to new prefix (e.g., "/mnt/user/" "/mnt/")')
    parser.add_argument('--migrate-tables', action='store_true',
                       help='Migrate model files from scanned_files to model_files table while preserving associations')
    parser.add_argument('--recover-missing', action='store_true',
                       help='Automatically recover missing files by checking alternative paths and destinations')
    parser.add_argument('--force-blake3', action='store_true',
                       help='Force rescan of non-SafeTensors files to add BLAKE3 hashes to database')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Reduce output verbosity')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output showing all operations and details')
    parser.add_argument('--version', action='version', version='Model Sorter v1.0.0')
    
    args = parser.parse_args()
    
    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"ERROR: Configuration file not found: {args.config}")
        print("Please create a config.ini file or specify the path with --config")
        sys.exit(1)
    
    # Initialize orchestrator
    orchestrator = ModelSorterOrchestrator(
        config_path=args.config,
        dry_run=args.dry_run,
        verbose=args.verbose or not args.quiet,  # Enable verbose if --verbose is set OR if --quiet is NOT set
        force_rescan=args.force_rescan,
        extract_metadata_limit=args.extract_metadata_limit,
        extract_metadata_all=args.extract_metadata_all,
        extract_metadata_all_rescan=args.extract_metadata_all_rescan,
        retry_failed=args.retry_failed,
        skip_folders=args.skip_folders,
        folder_limit=args.folder_limit
    )
    
    try:
        success = False
        
        # Auto-detect step based on flags
        if not args.step and (args.extract_metadata_limit or args.extract_metadata_all or args.extract_metadata_all_rescan):
            # If extract-metadata flags are provided without explicit step, run extract-metadata
            args.step = 'extract-metadata'
            print("Auto-detected step: extract-metadata (based on --extract-metadata flags)")
        elif not args.step and args.retry_failed:
            # If retry-failed flag is provided without explicit step, run retry-failed
            args.step = 'retry-failed'
            print("Auto-detected step: retry-failed (based on --retry-failed flag)")
        elif not args.step and args.migrate_paths:
            # If migrate-paths flag is provided without explicit step, run migrate-paths
            args.step = 'migrate-paths'
            print("Auto-detected step: migrate-paths (based on --migrate-paths flag)")
        elif not args.step and args.recover_missing:
            # If recover-missing flag is provided without explicit step, run recover-missing
            args.step = 'recover-missing'
            print("Auto-detected step: recover-missing (based on --recover-missing flag)")
        elif not args.step and args.force_blake3:
            # If force-blake3 flag is provided without explicit step, run force-blake3
            args.step = 'force-blake3'
            print("Auto-detected step: force-blake3 (based on --force-blake3 flag)")
        elif not args.step and args.update_tables:
            # If update-tables flag is provided without explicit step, run update-tables
            args.step = 'update-tables'
            print("Auto-detected step: update-tables (based on --update-tables flag)")
        elif not args.step and args.rescan_linked:
            # If rescan-linked flag is provided without explicit step, run rescan-linked
            args.step = 'rescan-linked'
            print("Auto-detected step: rescan-linked (based on --rescan-linked flag)")
        elif not args.step and args.rescan_linked_rebuild:
            # If rescan-linked-rebuild flag is provided without explicit step, run rescan-linked-rebuild
            args.step = 'rescan-linked-rebuild'
            print("Auto-detected step: rescan-linked-rebuild (based on --rescan-linked-rebuild flag)")
        elif not args.step and args.rebuild_links:
            # If rebuild-links flag is provided without explicit step, run rebuild-links (deprecated)
            args.step = 'rebuild-links'
            print("Auto-detected step: rebuild-links (based on --rebuild-links flag) [DEPRECATED: use --rescan-linked-rebuild]")
        elif not args.step and args.rescan_and_repair_metadata_text:
            # If rescan-and-repair-metadata-text flag is provided without explicit step
            args.step = 'rescan-and-repair-metadata-text'
            print("Auto-detected step: rescan-and-repair-metadata-text (based on --rescan-and-repair-metadata-text flag)")
        elif not args.step and args.migrate_tables:
            # If migrate-tables flag is provided without explicit step
            args.step = 'migrate-tables'
            print("Auto-detected step: migrate-tables (based on --migrate-tables flag)")
        
        if args.step:
            # Special handling for steps that require arguments
            if args.step == 'migrate-paths':
                if not args.migrate_paths:
                    print("ERROR: --migrate-paths step requires --migrate-paths OLD_PREFIX NEW_PREFIX arguments")
                    sys.exit(1)
                old_prefix, new_prefix = args.migrate_paths
                success = orchestrator.step_migrate_paths(old_prefix, new_prefix)
            else:
                # Run individual step
                success = orchestrator.run_individual_step(args.step)
        else:
            # Run full workflow
            success = orchestrator.run_full_workflow()
        
        if success:
            print("\nProcessing completed successfully!")
            sys.exit(0)
        else:
            print("\nProcessing completed with errors!")
            sys.exit(1)
            
    except BrokenPipeError:
        # Handle broken pipe error (e.g., when output is piped to head)
        # Suppress the error and exit gracefully
        try:
            sys.stdout.close()
        except:
            pass
        try:
            sys.stderr.close() 
        except:
            pass
        sys.exit(0)
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        sys.exit(1)
    finally:
        orchestrator.close()


if __name__ == "__main__":
    main()