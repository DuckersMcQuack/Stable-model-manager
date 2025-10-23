#!/usr/bin/env python3
"""
Sorting and Moving Logic
Sorts models into loras/base_model/model_name/ structure with associated files
"""

import json
import os
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import configparser
import re


class FileMover:
    """Handles file moving operations with safety checks"""
    
    def __init__(self, destination_root: str, dry_run: bool = False):
        self.destination_root = destination_root
        self.dry_run = dry_run
        
    def sanitize_folder_name(self, name: str, max_length: int = 100) -> str:
        """Sanitize folder name by removing invalid characters"""
        if not name:
            return "Unknown"
        
        # Replace invalid characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '', name)
        sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', sanitized)  # Remove control chars
        
        # Replace multiple spaces/dots with single ones
        sanitized = re.sub(r'\s+', ' ', sanitized)
        sanitized = re.sub(r'\.+', '.', sanitized)
        
        # Remove leading/trailing whitespace and dots
        sanitized = sanitized.strip(' .')
        
        # Truncate if too long
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length].rstrip(' .')
        
        # Fallback for empty names
        if not sanitized:
            sanitized = "Unknown"
        
        return sanitized
    
    def ensure_directory(self, directory: str) -> bool:
        """Ensure directory exists, create if needed"""
        try:
            if not self.dry_run:
                os.makedirs(directory, exist_ok=True)
            return True
        except Exception as e:
            print(f"Error creating directory {directory}: {e}")
            return False
    
    def move_file(self, source_path: str, target_path: str, 
                  overwrite: bool = False) -> Tuple[bool, str]:
        """Move file from source to target path"""
        try:
            # Ensure target directory exists
            target_dir = os.path.dirname(target_path)
            if not self.ensure_directory(target_dir):
                return False, f"Could not create target directory: {target_dir}"
            
            # Check if target exists
            if os.path.exists(target_path) and not overwrite:
                # Generate unique name
                base, ext = os.path.splitext(target_path)
                counter = 1
                while os.path.exists(f"{base}_{counter}{ext}"):
                    counter += 1
                target_path = f"{base}_{counter}{ext}"
            
            if self.dry_run:
                print(f"DRY RUN: Would move {source_path} -> {target_path}")
                return True, target_path
            else:
                # Perform the move
                shutil.move(source_path, target_path)
                print(f"Moved: {source_path} -> {target_path}")
                return True, target_path
        
        except Exception as e:
            error_msg = f"Error moving {source_path} to {target_path}: {e}"
            print(error_msg)
            return False, error_msg


class ModelSorter:
    """Main model sorting and organizing class"""
    
    def __init__(self, config_path: str = "config.ini", dry_run: bool = False):
        self.config = self._load_config(config_path)
        self.dry_run = dry_run
        self.db_path = "model_sorter.sqlite"
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        
        self.file_mover = FileMover(self.config['destination_directory'], dry_run)
        
        # Stats
        self.stats = {
            'models_moved': 0,
            'duplicates_moved': 0,
            'associated_files_moved': 0,
            'errors': 0
        }
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from INI file"""
        config = configparser.ConfigParser()
        config.read(config_path)
        
        return {
            'destination_directory': config.get('Paths', 'destination_directory', fallback='sorted-models'),
            'model_type_folders': config.get('Sorting', 'model_type_folders', 
                                           fallback='checkpoint,loras,embedding,textualinversion,hypernetwork,controlnet,other').split(','),
            'use_model_type_subfolders': config.getboolean('Sorting', 'use_model_type_subfolders', fallback=True),
            'sanitize_folder_names': config.getboolean('Sorting', 'sanitize_folder_names', fallback=True),
            'max_folder_name_length': config.getint('Sorting', 'max_folder_name_length', fallback=100),
            'skip_existing_files': config.getboolean('Processing', 'skip_existing_files', fallback=True),
            'extract_related_folders': config.getboolean('Processing', 'extract_related_folders', fallback=True)
        }
    
    def determine_model_folder_structure(self, model_info: Dict) -> Tuple[str, str]:
        """Determine the target folder structure for a model"""
        base_model = model_info.get('base_model', 'Other')
        model_type = model_info.get('model_type', 'LORA').upper()
        model_name = model_info.get('model_name', 'Unknown')
        
        # Sanitize names if configured
        if self.config['sanitize_folder_names']:
            base_model = self.file_mover.sanitize_folder_name(base_model, self.config['max_folder_name_length'])
            model_name = self.file_mover.sanitize_folder_name(model_name, self.config['max_folder_name_length'])
        
        # Determine model type folder
        type_folder = 'loras'  # Default for LORA and similar types
        if model_type in ['CHECKPOINT']:
            type_folder = 'checkpoint'
        elif model_type in ['EMBEDDING', 'TEXTUALINVERSION']:
            type_folder = 'embedding'
        elif model_type in ['HYPERNETWORK']:
            type_folder = 'hypernetwork'
        elif model_type in ['CONTROLNET']:
            type_folder = 'controlnet'
        elif model_type in ['VAE']:
            type_folder = 'vae'
        elif model_type in ['UPSCALER']:
            type_folder = 'upscaler'
        elif model_type in ['DORA']:
            type_folder = 'loras'  # DoRA goes in loras folder
        elif model_type in ['LOCON']:
            type_folder = 'loras'  # LoCon goes in loras folder
        elif model_type in ['MOTIONMODULE']:
            type_folder = 'motionmodule'
        elif model_type in ['AESTHETICGRADIENT']:
            type_folder = 'other'
        elif model_type in ['DETECTION']:
            type_folder = 'other'
        elif model_type in ['POSES']:
            type_folder = 'poses'
        elif model_type in ['WILDCARDS']:
            type_folder = 'wildcards'
        elif model_type in ['WORKFLOWS']:
            type_folder = 'workflows'
        elif model_type in ['OTHER'] or model_type not in ['LORA']:
            type_folder = 'other'
        
        # Build folder structure
        if self.config['use_model_type_subfolders']:
            folder_path = os.path.join(type_folder, base_model, model_name)
        else:
            folder_path = os.path.join(base_model, model_name)
        
        target_directory = os.path.join(self.config['destination_directory'], folder_path)
        
        return target_directory, folder_path
    
    def get_associated_files_for_model(self, model_path: str, model_name: str) -> List[str]:
        """Get all files associated with a model"""
        associated_files = []
        model_dir = os.path.dirname(model_path)
        model_basename = os.path.splitext(os.path.basename(model_path))[0]
        
        # Common extensions for associated files
        associated_extensions = [
            '.civitai.info', '.metadata.json', '.txt', '.yaml', '.yml', '.json',
            '.png', '.jpg', '.jpeg', '.webp', '.gif', '.preview.png', '.preview.jpg',
            '.preview.jpeg', '.preview.webp'
        ]
        
        # Look for files with same base name
        for ext in associated_extensions:
            candidate_path = os.path.join(model_dir, f"{model_basename}{ext}")
            if os.path.exists(candidate_path):
                associated_files.append(candidate_path)
        
        # Look in subdirectories if configured
        if self.config['extract_related_folders']:
            folder_patterns = ['_images', '_files', '_data', '_docs', '_samples', 
                             '-images', '-files', '-data', '-docs', '-samples',
                             'images', 'files', 'data', 'docs', 'samples']
            
            for pattern in folder_patterns:
                # Check for exact pattern match
                subfolder_path = os.path.join(model_dir, f"{model_basename}{pattern}")
                if os.path.isdir(subfolder_path):
                    # Add all files from subfolder
                    for item in os.listdir(subfolder_path):
                        item_path = os.path.join(subfolder_path, item)
                        if os.path.isfile(item_path):
                            associated_files.append(item_path)
                
                # Check for pattern in folder names
                for item in os.listdir(model_dir):
                    item_path = os.path.join(model_dir, item)
                    if os.path.isdir(item_path) and (pattern in item.lower() or model_basename.lower() in item.lower()):
                        for subitem in os.listdir(item_path):
                            subitem_path = os.path.join(item_path, subitem)
                            if os.path.isfile(subitem_path):
                                associated_files.append(subitem_path)
        
        return list(set(associated_files))  # Remove duplicates
    
    def _get_next_duplicate_number(self, model_name: str) -> int:
        """Get the next available duplicate number for a model"""
        cursor = self.conn.cursor()
        
        # Check existing duplicates folder for this model
        duplicates_base_path = os.path.join(self.config['destination_directory'], 'loras', 'duplicates')
        
        # Find the highest existing duplicate number
        highest_num = 0
        if os.path.exists(duplicates_base_path):
            for item in os.listdir(duplicates_base_path):
                if item.startswith(f"{model_name}_") and os.path.isdir(os.path.join(duplicates_base_path, item)):
                    try:
                        # Extract number from folder name
                        num_str = item.split('_')[-1]
                        num = int(num_str)
                        highest_num = max(highest_num, num)
                    except (ValueError, IndexError):
                        continue
        
        return highest_num + 1
    
    def move_model_and_associated_files(self, model_info: Dict) -> bool:
        """Move a model and all its associated files"""
        try:
            source_path = model_info['source_path']
            model_name = model_info['model_name']
            
            # Check if source file exists before attempting to move
            if not os.path.exists(source_path):
                print(f"Source file does not exist, checking for smart recovery: {source_path}")
                
                # Determine where the file should be and check if it's already there
                target_directory, folder_path = self.determine_model_folder_structure(model_info)
                model_filename = os.path.basename(source_path)
                target_model_path = os.path.join(target_directory, model_filename)
                
                if os.path.exists(target_model_path):
                    print(f"Model already exists at target, moving remaining associated files: {target_model_path}")
                    
                    # Find and move any remaining associated files
                    associated_files = self.get_associated_files_for_model(source_path, model_name)
                    moved_count = 0
                    
                    for assoc_file in associated_files:
                        if os.path.exists(assoc_file):
                            assoc_filename = os.path.basename(assoc_file)
                            target_assoc_path = os.path.join(target_directory, assoc_filename)
                            
                            # Only move if target doesn't exist
                            if not os.path.exists(target_assoc_path):
                                assoc_success, assoc_final_path = self.file_mover.move_file(assoc_file, target_assoc_path)
                                if assoc_success:
                                    moved_count += 1
                                    self.stats['associated_files_moved'] += 1
                                    print(f"  Moved remaining file: {assoc_file} -> {assoc_final_path}")
                    
                    print(f"Smart recovery: moved {moved_count} remaining associated files")
                    
                    # Update database to mark as already moved
                    cursor = self.conn.cursor()
                    cursor.execute('''
                        UPDATE model_files 
                        SET target_path = ?, status = 'moved', updated_at = ?
                        WHERE source_path = ?
                    ''', (target_model_path, int(time.time()), source_path))
                    self.conn.commit()
                    return True
                else:
                    print(f"Model not found at expected target location: {target_model_path}")
                
                # Update database to mark as missing
                cursor = self.conn.cursor()
                cursor.execute('''
                    UPDATE model_files 
                    SET status = 'file_missing', updated_at = ?
                    WHERE source_path = ?
                ''', (int(time.time()), source_path))
                self.conn.commit()
                return False
            
            # Determine target structure
            target_directory, folder_path = self.determine_model_folder_structure(model_info)
            
            # Get model filename
            model_filename = os.path.basename(source_path)
            target_model_path = os.path.join(target_directory, model_filename)
            
            # Check if target already exists - treat as duplicate instead of skipping
            if os.path.exists(target_model_path):
                print(f"Target already exists for {source_path}, moving to duplicates folder instead")
                # Convert to duplicate and move to duplicates folder
                duplicate_info = {
                    'source_path': source_path,
                    'model_name': model_name,
                    'base_model': model_info.get('base_model', 'Unknown'),
                    'model_type': model_info.get('model_type', 'LORA'),
                    'duplicate_number': self._get_next_duplicate_number(model_name)
                }
                return self.move_duplicate_model(duplicate_info)
            
            # Move the model file
            success, final_path = self.file_mover.move_file(source_path, target_model_path)
            if not success:
                self.stats['errors'] += 1
                return False
            
            # Update database with new path
            cursor = self.conn.cursor()
            cursor.execute('''
                UPDATE model_files 
                SET target_path = ?, status = 'moved', updated_at = ?
                WHERE source_path = ?
            ''', (final_path, int(time.time()), source_path))
            
            self.stats['models_moved'] += 1
            
            # Get and move associated files
            associated_files = self.get_associated_files_for_model(source_path, model_name)
            
            for assoc_file in associated_files:
                if os.path.exists(assoc_file):  # File might have been moved already
                    assoc_filename = os.path.basename(assoc_file)
                    target_assoc_path = os.path.join(target_directory, assoc_filename)
                    
                    assoc_success, assoc_final_path = self.file_mover.move_file(assoc_file, target_assoc_path)
                    if assoc_success:
                        self.stats['associated_files_moved'] += 1
                        
                        # Log associated file move - determine association type
                        assoc_type = 'other'
                        if '.civitai.info' in assoc_filename:
                            assoc_type = 'civitai_info'
                        elif '.metadata.json' in assoc_filename:
                            assoc_type = 'metadata'
                        elif any(ext in assoc_filename.lower() for ext in ['.png', '.jpg', '.jpeg', '.webp', 'preview']):
                            assoc_type = 'image'
                        
                        # Get model file ID
                        cursor.execute('SELECT id FROM model_files WHERE source_path = ?', (source_path,))
                        model_file_row = cursor.fetchone()
                        if model_file_row:
                            model_file_id = model_file_row[0]
                            
                            # After file has been moved, look up using the target path
                            lookup_path = assoc_final_path if assoc_success else assoc_file
                            print(f"🔍 Looking up database record for: {lookup_path}")
                            
                            # Get scanned file ID for associated file (may be None for non-model files)
                            # Try both the current path and server path format for compatibility
                            cursor.execute('SELECT id FROM scanned_files WHERE file_path = ?', (lookup_path,))
                            scanned_file_row = cursor.fetchone()
                            
                            # If not found and path starts with /mnt/, also try /mnt/user/ variant
                            if not scanned_file_row and lookup_path.startswith('/mnt/'):
                                server_path = lookup_path.replace('/mnt/', '/mnt/user/', 1)
                                cursor.execute('SELECT id FROM scanned_files WHERE file_path = ?', (server_path,))
                                scanned_file_row = cursor.fetchone()
                                print(f"🔍 Tried server path variant: {server_path} -> {'Found' if scanned_file_row else 'Not found'}")
                            
                            # If still not found, check if file exists but not in database yet
                            if not scanned_file_row and assoc_success:
                                assoc_filename = os.path.basename(lookup_path)
                                print(f"🔍 File moved successfully but not in database, adding: {assoc_filename}")
                                
                                # Add the moved file to scanned_files database
                                if os.path.exists(lookup_path):
                                    stat = os.stat(lookup_path)
                                    file_size = stat.st_size
                                    last_modified = int(stat.st_mtime)
                                    file_name = os.path.basename(lookup_path)
                                    
                                    # Get file extension
                                    _, extension = os.path.splitext(file_name)
                                    
                                    # Determine file type
                                    if '.metadata.json' in lookup_path:
                                        file_type = 'text'
                                    elif '.civitai.info' in lookup_path:
                                        file_type = 'text'
                                    elif any(ext in lookup_path.lower() for ext in ['.png', '.jpg', '.jpeg', '.webp']):
                                        file_type = 'image'
                                    else:
                                        file_type = 'text'
                                    
                                    # Insert into scanned_files with the target path
                                    cursor.execute('''
                                        INSERT INTO scanned_files 
                                        (file_path, file_name, file_size, file_type, extension, last_modified, scan_date, sha256, autov3, is_processed)
                                        VALUES (?, ?, ?, ?, ?, ?, ?, '', '', 0)
                                    ''', (lookup_path, file_name, file_size, file_type, extension, last_modified, int(time.time())))
                                    
                                    # Get the new scanned_file_id
                                    scanned_file_id = cursor.lastrowid
                                    scanned_file_row = (scanned_file_id,)
                                    print(f"✅ Added to database with ID: {scanned_file_id}")
                            
                            # If still not found, try filename-based search as fallback
                            if not scanned_file_row:
                                assoc_filename = os.path.basename(assoc_file)
                                cursor.execute('''
                                    SELECT id, file_path FROM scanned_files 
                                    WHERE file_path LIKE ? 
                                    ORDER BY file_path DESC 
                                    LIMIT 1
                                ''', (f"%{assoc_filename}",))
                                potential_match = cursor.fetchone()
                                
                                if potential_match:
                                    scanned_file_id, found_path = potential_match
                                    if os.path.exists(found_path):
                                        print(f"📍 Found similar file in database: {found_path}")
                                        scanned_file_row = (scanned_file_id,)
                            

                            
                            # Only insert if we have a valid scanned_file_id (file was in our scan)
                            if scanned_file_row:
                                scanned_file_id = scanned_file_row[0]
                                print(f"✅ Creating association record:")
                                print(f"   Model ID: {model_file_id}")
                                print(f"   Scanned File ID: {scanned_file_id}")
                                print(f"   Source: {assoc_file}")
                                print(f"   Target: {assoc_final_path}")
                                cursor.execute('''
                                    INSERT OR REPLACE INTO associated_files 
                                    (model_file_id, scanned_file_id, association_type, source_path, target_path, is_moved)
                                    VALUES (?, ?, ?, ?, ?, 1)
                                ''', (model_file_id, scanned_file_id, assoc_type, assoc_file, assoc_final_path))
                            else:
                                # Associated file wasn't in our scan (probably created during metadata extraction)
                                print(f"❌ Associated file not in scan database, skipping DB record: {assoc_file}")
                                print(f"   File moved successfully: {assoc_success}")
                                print(f"   Target path: {assoc_final_path if assoc_success else 'N/A'}")
                                print(f"   Target exists: {os.path.exists(assoc_final_path) if assoc_success else 'N/A'}")
                    else:
                        self.stats['errors'] += 1
            
            self.conn.commit()
            return True
            
        except Exception as e:
            print(f"Error moving model {model_info.get('source_path', 'Unknown')}: {e}")
            self.stats['errors'] += 1
            return False
    
    def move_duplicate_model(self, duplicate_info: Dict) -> bool:
        """Move a duplicate model to the duplicates folder"""
        try:
            source_path = duplicate_info['source_path']
            model_name = duplicate_info['model_name']
            duplicate_number = duplicate_info['duplicate_number']
            
            # Check if source file exists before attempting to move
            if not os.path.exists(source_path):
                print(f"Source file does not exist, checking for smart recovery: {source_path}")
                
                # Check if the model was already moved to target location
                duplicate_folder_name = f"{model_name}_{duplicate_number}"
                target_directory = os.path.join(self.config['destination_directory'], 'loras', 'duplicates', duplicate_folder_name)
                model_filename = os.path.basename(source_path)
                target_model_path = os.path.join(target_directory, model_filename)
                
                if os.path.exists(target_model_path):
                    print(f"Model already exists at target, moving remaining associated files: {target_model_path}")
                    
                    # Find and move any remaining associated files
                    associated_files = self.get_associated_files_for_model(source_path, model_name)
                    moved_count = 0
                    
                    for assoc_file in associated_files:
                        if os.path.exists(assoc_file):
                            assoc_filename = os.path.basename(assoc_file)
                            target_assoc_path = os.path.join(target_directory, assoc_filename)
                            
                            # Only move if target doesn't exist
                            if not os.path.exists(target_assoc_path):
                                assoc_success, assoc_final_path = self.file_mover.move_file(assoc_file, target_assoc_path)
                                if assoc_success:
                                    moved_count += 1
                                    self.stats['associated_files_moved'] += 1
                                    print(f"  Moved remaining file: {assoc_file} -> {assoc_final_path}")
                    
                    print(f"Smart recovery: moved {moved_count} remaining associated files")
                    
                    # Update database to mark as already moved
                    cursor = self.conn.cursor()
                    cursor.execute('''
                        UPDATE model_files 
                        SET target_path = ?, status = 'duplicate_moved', updated_at = ?
                        WHERE source_path = ?
                    ''', (target_model_path, int(time.time()), source_path))
                    self.conn.commit()
                    return True
                else:
                    print(f"Model not found at expected target location: {target_model_path}")
                
                # Update database to mark as missing
                cursor = self.conn.cursor()
                cursor.execute('''
                    UPDATE model_files 
                    SET status = 'file_missing', updated_at = ?
                    WHERE source_path = ?
                ''', (int(time.time()), source_path))
                self.conn.commit()
                return False
            
            # Create duplicate folder name
            duplicate_folder_name = f"{model_name}_{duplicate_number}"
            target_directory = os.path.join(self.config['destination_directory'], 'loras', 'duplicates', duplicate_folder_name)
            
            # Move model file
            model_filename = os.path.basename(source_path)
            target_model_path = os.path.join(target_directory, model_filename)
            
            success, final_path = self.file_mover.move_file(source_path, target_model_path)
            if not success:
                self.stats['errors'] += 1
                return False
            
            # Update database
            cursor = self.conn.cursor()
            cursor.execute('''
                UPDATE model_files 
                SET target_path = ?, status = 'duplicate_moved', updated_at = ?
                WHERE source_path = ?
            ''', (final_path, int(time.time()), source_path))
            
            self.stats['duplicates_moved'] += 1
            
            # Move associated files
            associated_files = self.get_associated_files_for_model(source_path, model_name)
            
            for assoc_file in associated_files:
                if os.path.exists(assoc_file):
                    assoc_filename = os.path.basename(assoc_file)
                    target_assoc_path = os.path.join(target_directory, assoc_filename)
                    
                    assoc_success, assoc_final_path = self.file_mover.move_file(assoc_file, target_assoc_path)
                    if assoc_success:
                        self.stats['associated_files_moved'] += 1
            
            self.conn.commit()
            return True
            
        except Exception as e:
            print(f"Error moving duplicate {duplicate_info.get('source_path', 'Unknown')}: {e}")
            self.stats['errors'] += 1
            return False
    
    def enhance_model_info_from_civitai(self, model: dict) -> dict:
        """Enhance model information using civitai.sqlite database and pattern detection"""
        try:
            civitai_db_path = os.path.join("Database", "civitai.sqlite")
            enhanced_model = dict(model)  # Copy original
            
            # First try civitai database lookup
            if os.path.exists(civitai_db_path):
                # Connect to civitai database
                civitai_conn = sqlite3.connect(civitai_db_path)
                civitai_conn.row_factory = sqlite3.Row
                civitai_cursor = civitai_conn.cursor()
                
                # Look up model by SHA256 hash
                sha256 = model.get('sha256')
                if sha256:
                    civitai_cursor.execute('''
                        SELECT 
                            m.name as civitai_name,
                            m.type as civitai_type,
                            json_extract(mf.data, '$.name') as filename,
                            json_extract(mf.data, '$.metadata.baseModel') as base_model
                        FROM models m
                        JOIN model_files mf ON m.id = mf.model_id
                        WHERE UPPER(mf.sha256) = UPPER(?)
                        AND mf.type = 'Model'
                        LIMIT 1
                    ''', (sha256,))
                    
                    civitai_info = civitai_cursor.fetchone()
                    if civitai_info:
                        # Use civitai name if current is None/empty
                        if not enhanced_model.get('model_name') or enhanced_model.get('model_name') == 'None':
                            enhanced_model['model_name'] = civitai_info['civitai_name']
                            print(f"📝 Enhanced model name: {civitai_info['civitai_name']}")
                        
                        # Use civitai base model if available
                        if civitai_info['base_model'] and (not enhanced_model.get('base_model') or enhanced_model.get('base_model') == 'Unknown'):
                            enhanced_model['base_model'] = civitai_info['base_model']
                            print(f"📝 Enhanced base model from civitai: {civitai_info['base_model']}")
                        
                        # Add civitai type information
                        enhanced_model['civitai_type'] = civitai_info['civitai_type']
                
                civitai_conn.close()
            
            # If still no base model or base model is "Unknown" or "Other", try pattern-based detection
            if not enhanced_model.get('base_model') or enhanced_model.get('base_model') in ['Unknown', 'Other']:
                detected_base_model = self._detect_base_model_from_patterns(enhanced_model)
                if detected_base_model:
                    enhanced_model['base_model'] = detected_base_model
                    print(f"🔍 Detected base model from patterns: {detected_base_model}")
                else:
                    # If pattern detection failed, try comprehensive associated file analysis
                    detected_from_files = self._detect_base_model_from_associated_files(enhanced_model)
                    if detected_from_files:
                        enhanced_model['base_model'] = detected_from_files
                        print(f"📄 Detected base model from associated files: {detected_from_files}")
            
            return enhanced_model
            
        except Exception as e:
            print(f"⚠️  Error enhancing model info: {e}")
        
        return model
    
    def _detect_base_model_from_patterns(self, model: dict) -> str | None:
        """Detect base model from filename and path patterns using dynamic pattern matching"""
        # Get various sources to check for patterns
        sources = []
        
        if model.get('model_name'):
            sources.append(model['model_name'])
        if model.get('source_path'):
            sources.append(model['source_path'])
            sources.append(os.path.basename(model['source_path']))
            # Also add filename without extension
            base_filename = os.path.splitext(os.path.basename(model['source_path']))[0]
            sources.append(base_filename)
        if model.get('file_name'):
            sources.append(model['file_name'])
            # Also add filename without extension
            base_filename = os.path.splitext(model['file_name'])[0]
            sources.append(base_filename)
        
        # Dynamic base model patterns - any separator between key components
        # Pattern structure: [prefix][separator][version][separator][suffix]
        # Separator can be: space, dot, dash, underscore, or nothing
        sep = r'[._\s-]*'  # Any combination of separators or none
        
        base_model_patterns = [
            # SDXL variants (most specific first)
            (rf'(?i)sdxl{sep}lightning|lightning{sep}sdxl', 'SDXL Lightning'),
            (rf'(?i)sdxl{sep}turbo|turbo{sep}sdxl', 'SDXL Turbo'),
            (rf'(?i)sdxl{sep}hyper|hyper{sep}sdxl', 'SDXL Hyper'),
            (rf'(?i)sdxl{sep}distilled|distilled{sep}sdxl', 'SDXL Distilled'),
            (rf'(?i)sdxl{sep}1{sep}0{sep}lcm|lcm{sep}sdxl', 'SDXL 1.0 LCM'),
            (rf'(?i)sdxl{sep}1{sep}0', 'SDXL 1.0'),
            (rf'(?i)sdxl{sep}0{sep}9', 'SDXL 0.9'),
            (rf'(?i)sdxl', 'SDXL 1.0'),
            
            # SD variants (most specific first)
            (rf'(?i)sd{sep}3{sep}5{sep}large{sep}turbo', 'SD 3.5 Large Turbo'),
            (rf'(?i)sd{sep}3{sep}5{sep}large', 'SD 3.5 Large'),
            (rf'(?i)sd{sep}3{sep}5{sep}medium', 'SD 3.5 Medium'),
            (rf'(?i)sd{sep}3{sep}5', 'SD 3.5'),
            (rf'(?i)sd{sep}3', 'SD 3'),
            (rf'(?i)sd{sep}2{sep}1{sep}768', 'SD 2.1 768'),
            (rf'(?i)sd{sep}2{sep}1{sep}unclip', 'SD 2.1 Unclip'),
            (rf'(?i)sd{sep}2{sep}1', 'SD 2.1'),
            (rf'(?i)sd{sep}2{sep}0{sep}768', 'SD 2.0 768'),
            (rf'(?i)sd{sep}2{sep}0', 'SD 2.0'),
            (rf'(?i)sd{sep}1{sep}5{sep}lcm|lcm{sep}sd{sep}1{sep}5', 'SD 1.5 LCM'),
            (rf'(?i)sd{sep}1{sep}5{sep}hyper|hyper{sep}sd{sep}1{sep}5', 'SD 1.5 Hyper'),
            (rf'(?i)sd{sep}1{sep}5', 'SD 1.5'),
            (rf'(?i)sd{sep}1{sep}4', 'SD 1.4'),
            
            # FLUX variants
            (rf'(?i)flux{sep}1{sep}kontext', 'Flux.1 Kontext'),
            (rf'(?i)flux{sep}1{sep}krea', 'Flux.1 Krea'),
            (rf'(?i)flux{sep}1{sep}d', 'Flux.1 D'),
            (rf'(?i)flux{sep}1{sep}s', 'Flux.1 S'),
            (rf'(?i)flux', 'Flux.1 D'),
            
            # Popular models
            (rf'(?i)illustrious', 'Illustrious'),
            (rf'(?i)pony{sep}xl|pony{sep}v6', 'Pony'),
            (rf'(?i)pony', 'Pony'),
            (rf'(?i)noob{sep}ai', 'NoobAI'),
            
            # Video models (most specific first)
            (rf'(?i)hunyuan{sep}video', 'Hunyuan Video'),
            
            # Wan Video variants (specific versions first)
            (rf'(?i)wan{sep}video{sep}14b{sep}t2v', 'Wan Video 14B t2v'),
            (rf'(?i)wan{sep}video{sep}14b{sep}i2v{sep}720p', 'Wan Video 14B i2v 720p'),
            (rf'(?i)wan{sep}video{sep}14b{sep}i2v{sep}480p', 'Wan Video 14B i2v 480p'),
            (rf'(?i)wan{sep}video{sep}2{sep}2{sep}t2v{sep}a14b', 'Wan Video 2.2 T2V-A14B'),
            (rf'(?i)wan{sep}video{sep}2{sep}2{sep}i2v{sep}a14b', 'Wan Video 2.2 I2V-A14B'),
            (rf'(?i)wan{sep}video{sep}2{sep}2{sep}ti2v{sep}5b', 'Wan Video 2.2 TI2V-5B'),
            (rf'(?i)wan{sep}video{sep}1{sep}3b{sep}t2v', 'Wan Video 1.3B t2v'),
            
            # Wan 2.2 filename patterns (detect from filename)
            (rf'(?i)wan{sep}?2\.?2.*?low.*?noise|wan{sep}?2\.?2.*?lownoise', 'Wan 2.2'),
            (rf'(?i)wan{sep}?2\.?2.*?high.*?noise|wan{sep}?2\.?2.*?highnoise', 'Wan 2.2'),
            (rf'(?i)wan{sep}?2\.?2.*?t2v', 'Wan 2.2'),
            (rf'(?i)wan{sep}?2\.?2.*?i2v', 'Wan 2.2'),
            (rf'(?i)wan{sep}?2\.?2', 'Wan 2.2'),
            (rf'(?i)wan2\.?2', 'Wan 2.2'),
            
            # Generic Wan patterns
            (rf'(?i)wan{sep}video', 'Wan Video'),
            (rf'(?i)wan{sep}', 'Wan 2.2'),
            
            # Other video models
            (rf'(?i)cog{sep}video{sep}x', 'CogVideoX'),
            (rf'(?i)svd{sep}xt', 'SVD XT'),
            (rf'(?i)svd', 'SVD'),
            (rf'(?i)mochi', 'Mochi'),
            (rf'(?i)ltxv', 'LTXV'),
            (rf'(?i)veo{sep}3', 'Veo 3'),
            
            # Other AI models
            (rf'(?i)hunyuan{sep}1', 'Hunyuan 1'),
            (rf'(?i)hidream', 'HiDream'),
            (rf'(?i)stable{sep}cascade', 'Stable Cascade'),
            (rf'(?i)pixart{sep}e', 'PixArt E'),
            (rf'(?i)pixart{sep}a', 'PixArt a'),
            (rf'(?i)lumina', 'Lumina'),
            (rf'(?i)kolors', 'Kolors'),
            (rf'(?i)playground{sep}v2', 'Playground v2'),
            (rf'(?i)chroma', 'Chroma'),
            (rf'(?i)auraflow', 'AuraFlow'),
            (rf'(?i)qwen', 'Qwen'),
            (rf'(?i)odor', 'ODOR'),
            (rf'(?i)openai', 'OpenAI'),
            (rf'(?i)nano{sep}banana', 'Nano Banana'),
            (rf'(?i)imagen{sep}4', 'Imagen4'),
            
            # Generic patterns (last resort)
            (rf'(?i)xl', 'SDXL 1.0'),  # Generic XL reference
            (rf'(?i)anime|nai|novelai', 'SD 1.5'),  # Common anime model patterns
        ]
        
        # Check each source against patterns
        for source in sources:
            if not source:
                continue
                
            for pattern, base_model in base_model_patterns:
                if re.search(pattern, source):
                    return base_model
        
        return None
    
    def _detect_base_model_from_associated_files(self, model: dict) -> str | None:
        """Detect base model by analyzing associated files (text, metadata, images) for clues"""
        if not model.get('source_path'):
            return None
            
        source_path = model['source_path']
        source_dir = os.path.dirname(source_path)
        base_name = os.path.splitext(os.path.basename(source_path))[0]
        
        print(f"🔍 Analyzing associated files for {base_name}...")
        
        # Define associated file patterns to look for
        associated_patterns = [
            f"{base_name}.txt",           # Training info files
            f"{base_name}.json",          # Metadata files
            f"{base_name}.yaml",          # Config files
            f"{base_name}.yml",           # Config files
            f"{base_name}.civitai.info",  # Civitai info files
            f"{base_name}.metadata.json", # Extended metadata
            "*.txt",                      # Any text files in same dir
            "*.json",                     # Any JSON files in same dir
            "*.yaml",                     # Any YAML files in same dir
            "*.yml",                      # Any YML files in same dir
            "*.md",                       # README or description files
            "*.readme",                   # README files
            "training_config.json",       # Common training config name
            "config.json",                # Generic config
            "model_info.txt",             # Model info files
        ]
        
        detected_info = []
        files_checked = 0
        
        try:
            # Check for specific associated files
            for pattern in associated_patterns:
                if '*' in pattern:
                    # Glob pattern - check all matching files in directory
                    import glob
                    matching_files = glob.glob(os.path.join(source_dir, pattern))
                    for file_path in matching_files[:5]:  # Limit to first 5 matches per pattern
                        if os.path.isfile(file_path):
                            info = self._analyze_file_content(file_path, base_name)
                            if info:
                                detected_info.extend(info)
                            files_checked += 1
                else:
                    # Exact file pattern
                    file_path = os.path.join(source_dir, pattern)
                    if os.path.isfile(file_path):
                        info = self._analyze_file_content(file_path, base_name)
                        if info:
                            detected_info.extend(info)
                        files_checked += 1
            
            # Also check parent directory for common config files
            parent_dir = os.path.dirname(source_dir)
            if parent_dir != source_dir:  # Avoid infinite loops
                for pattern in ["*.txt", "*.json", "*.md"]:
                    import glob
                    matching_files = glob.glob(os.path.join(parent_dir, pattern))
                    for file_path in matching_files[:3]:  # Limit parent dir checks
                        if os.path.isfile(file_path):
                            info = self._analyze_file_content(file_path, base_name)
                            if info:
                                detected_info.extend(info)
                            files_checked += 1
            
            print(f"   Checked {files_checked} associated files")
            
            if detected_info:
                # Analyze detected information to determine most likely base model
                return self._determine_base_model_from_clues(detected_info)
            else:
                print(f"   No base model clues found in associated files")
                return None
                
        except Exception as e:
            print(f"   Error analyzing associated files: {e}")
            return None
    
    def _analyze_file_content(self, file_path: str, model_name: str) -> list:
        """Analyze individual file content for base model clues"""
        clues = []
        
        try:
            # Try to read file with different encodings
            content = ""
            for encoding in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read().lower()
                    break
                except UnicodeDecodeError:
                    continue
            
            if not content:
                return clues
            
            # File size check - skip very large files
            if len(content) > 100000:  # 100KB limit
                print(f"   Skipping large file: {os.path.basename(file_path)}")
                return clues
            
            print(f"   Analyzing: {os.path.basename(file_path)}")
            
            # Look for base model indicators in content
            base_model_indicators = [
                # Direct mentions
                (r'(?i)base[_\s]*model[:\s]*["\']?([^"\';\s\n]+)', 'base_model_field'),
                (r'(?i)model[_\s]*version[:\s]*["\']?([^"\';\s\n]+)', 'model_version'),
                (r'(?i)stable[_\s]*diffusion[_\s]*([0-9\.]+)', 'sd_version'),
                (r'(?i)sdxl[_\s]*([0-9\.]*)', 'sdxl_version'),
                (r'(?i)flux[_\s]*([0-9\.]*)', 'flux_version'),
                
                # Training/model type keywords
                (r'(?i)trained[_\s]*on[:\s]*["\']?([^"\';\s\n]+)', 'trained_on'),
                (r'(?i)checkpoint[_\s]*type[:\s]*["\']?([^"\';\s\n]+)', 'checkpoint_type'),
                (r'(?i)architecture[:\s]*["\']?([^"\';\s\n]+)', 'architecture'),
                
                # Specific model patterns
                (r'(?i)(sd[_\s]*1[._\s]*5|stable[_\s]*diffusion[_\s]*1[._\s]*5)', 'sd_15_pattern'),
                (r'(?i)(sd[_\s]*2[._\s]*[01]|stable[_\s]*diffusion[_\s]*2[._\s]*[01])', 'sd_2_pattern'),
                (r'(?i)(sdxl|stable[_\s]*diffusion[_\s]*xl)', 'sdxl_pattern'),
                (r'(?i)(flux[._\s]*1|flux[_\s]*dev)', 'flux_pattern'),
                (r'(?i)(pony[_\s]*xl|pony[_\s]*v6)', 'pony_pattern'),
                
                # Resolution indicators (can hint at model type)
                (r'(?i)resolution[:\s]*["\']?(\d+x?\d*)', 'resolution'),
                (r'(?i)(\d{3,4})\s*x\s*(\d{3,4})', 'resolution_pattern'),
            ]
            
            for pattern, clue_type in base_model_indicators:
                matches = re.finditer(pattern, content)
                for match in matches:
                    clue_value = match.group(1) if match.groups() else match.group(0)
                    clues.append({
                        'type': clue_type,
                        'value': clue_value.strip(),
                        'source': os.path.basename(file_path),
                        'confidence': self._calculate_clue_confidence(clue_type, clue_value)
                    })
            
            # Look for common model names/references
            model_name_patterns = [
                'stable diffusion', 'sd 1.5', 'sd 2.0', 'sd 2.1', 'sdxl', 'flux',
                'pony', 'illustrious', 'noobai', 'anime', 'realistic'
            ]
            
            for pattern in model_name_patterns:
                if pattern in content:
                    clues.append({
                        'type': 'model_reference',
                        'value': pattern,
                        'source': os.path.basename(file_path),
                        'confidence': 0.6
                    })
            
        except Exception as e:
            print(f"   Error reading {os.path.basename(file_path)}: {e}")
        
        return clues
    
    def _calculate_clue_confidence(self, clue_type: str, clue_value: str) -> float:
        """Calculate confidence score for a clue based on type and value"""
        # Base confidences by clue type
        base_confidences = {
            'base_model_field': 0.9,      # Direct base_model field
            'model_version': 0.8,         # Model version field
            'sd_version': 0.85,           # SD version pattern
            'sdxl_version': 0.85,         # SDXL version pattern
            'flux_version': 0.85,         # Flux version pattern
            'trained_on': 0.7,            # Trained on field
            'checkpoint_type': 0.75,      # Checkpoint type
            'architecture': 0.7,          # Architecture field
            'sd_15_pattern': 0.8,         # SD 1.5 pattern
            'sd_2_pattern': 0.8,          # SD 2.x pattern
            'sdxl_pattern': 0.8,          # SDXL pattern
            'flux_pattern': 0.8,          # Flux pattern
            'pony_pattern': 0.8,          # Pony pattern
            'resolution': 0.5,            # Resolution hint
            'resolution_pattern': 0.5,    # Resolution pattern
            'model_reference': 0.6,       # General model reference
        }
        
        confidence = base_confidences.get(clue_type, 0.3)
        
        # Adjust confidence based on value quality
        if clue_value:
            value_lower = clue_value.lower()
            if 'sd' in value_lower or 'stable' in value_lower or 'diffusion' in value_lower:
                confidence += 0.1
            if any(version in value_lower for version in ['1.5', '2.0', '2.1', 'xl']):
                confidence += 0.1
            if len(clue_value.strip()) < 3:  # Very short values are less reliable
                confidence -= 0.2
        
        return min(1.0, max(0.0, confidence))
    
    def _determine_base_model_from_clues(self, clues: list) -> str | None:
        """Determine most likely base model from collected clues"""
        if not clues:
            return None
        
        print(f"   Found {len(clues)} clues, analyzing...")
        
        # Group clues by potential base model
        model_scores = {}
        
        for clue in clues:
            value = clue['value'].lower()
            confidence = clue['confidence']
            
            # Map clue values to base models
            if any(pattern in value for pattern in ['1.5', '1_5', 'sd15']):
                model_scores['SD 1.5'] = model_scores.get('SD 1.5', 0) + confidence
            elif any(pattern in value for pattern in ['2.0', '2_0', 'sd20']):
                model_scores['SD 2.0'] = model_scores.get('SD 2.0', 0) + confidence
            elif any(pattern in value for pattern in ['2.1', '2_1', 'sd21']):
                model_scores['SD 2.1'] = model_scores.get('SD 2.1', 0) + confidence
            elif any(pattern in value for pattern in ['xl', 'sdxl']):
                model_scores['SDXL 1.0'] = model_scores.get('SDXL 1.0', 0) + confidence
            elif 'flux' in value:
                model_scores['Flux.1 D'] = model_scores.get('Flux.1 D', 0) + confidence
            elif 'pony' in value:
                model_scores['Pony'] = model_scores.get('Pony', 0) + confidence
            elif 'illustrious' in value:
                model_scores['Illustrious'] = model_scores.get('Illustrious', 0) + confidence
            elif any(pattern in value for pattern in ['512', '768']) and 'sd' in value:
                # Resolution hints for SD versions
                if '768' in value:
                    model_scores['SD 2.1 768'] = model_scores.get('SD 2.1 768', 0) + confidence * 0.7
                else:
                    model_scores['SD 1.5'] = model_scores.get('SD 1.5', 0) + confidence * 0.7
            elif 'stable diffusion' in value or 'sd' in value:
                # Generic SD reference - default to 1.5
                model_scores['SD 1.5'] = model_scores.get('SD 1.5', 0) + confidence * 0.5
        
        if model_scores:
            # Return the base model with highest confidence score
            best_match = max(model_scores.items(), key=lambda x: x[1])
            confidence_threshold = 0.7  # Minimum confidence to accept detection
            
            if best_match[1] >= confidence_threshold:
                print(f"   🎯 Detected base model: {best_match[0]} (confidence: {best_match[1]:.2f})")
                return best_match[0]
            else:
                print(f"   ⚠️  Best match {best_match[0]} below confidence threshold ({best_match[1]:.2f} < {confidence_threshold})")
        
        return None
    
    def log_unmatched_models(self, unmatched_models: list) -> None:
        """Log unmatched models to unknown_models.json for analysis"""
        if not unmatched_models:
            return
            
        # Group by unique patterns to avoid duplicates
        unique_patterns = {}
        for model in unmatched_models:
            # Create a key based on model name and path components
            key_parts = []
            if model.get('model_name'):
                key_parts.append(model['model_name'])
            if model.get('source_path'):
                # Extract meaningful path components
                path_parts = model['source_path'].split('/')
                relevant_parts = [p for p in path_parts if p and not p.startswith('mnt') and not p.lower().startswith('archive')]
                key_parts.extend(relevant_parts[:3])  # First 3 relevant parts
            
            key = ' | '.join(key_parts)
            if key not in unique_patterns:
                unique_patterns[key] = {
                    'model_name': model.get('model_name', 'Unknown'),
                    'source_path': model.get('source_path', 'Unknown'),
                    'file_name': model.get('file_name', 'Unknown'),
                    'current_base_model': model.get('base_model', 'Unknown'),
                    'analysis': {
                        'path_contains': self._analyze_path_components(model.get('source_path', '')),
                        'filename_contains': self._analyze_filename_components(model.get('file_name', '')),
                        'suggested_patterns': self._suggest_patterns(model)
                    }
                }
        
        output_file = 'unknown_models.json'
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'analysis_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'total_unmatched': len(unique_patterns),
                    'unmatched_models': unique_patterns,
                    'note': 'These models could not be matched to base model patterns. Analyze the path/filename components to improve pattern detection.'
                }, f, indent=2, ensure_ascii=False)
            
            print(f"📝 Logged {len(unique_patterns)} unique unmatched models to {output_file}")
            
        except Exception as e:
            print(f"⚠️  Error writing unmatched models log: {e}")
    
    def _analyze_path_components(self, path: str) -> list:
        """Extract potential base model indicators from path"""
        if not path:
            return []
        
        components = []
        # Split path and analyze each part
        parts = path.replace('\\', '/').split('/')
        for part in parts:
            if part and len(part) > 1:
                # Look for version-like patterns
                if re.search(r'\d+\.?\d*', part):
                    components.append(part)
                # Look for model-like keywords
                elif re.search(r'(?i)(sd|sdxl|flux|pony|illustrious|xl|v\d)', part):
                    components.append(part)
        
        return components[:5]  # Limit to first 5 relevant components
    
    def _analyze_filename_components(self, filename: str) -> list:
        """Extract potential base model indicators from filename"""
        if not filename:
            return []
        
        components = []
        # Remove extension and analyze
        name = os.path.splitext(filename)[0]
        
        # Look for version patterns
        version_matches = re.findall(r'(?i)(sd|sdxl|flux|pony|v\d+\.?\d*|xl)', name)
        components.extend(version_matches)
        
        return components[:3]  # Limit to first 3 relevant components
    
    def _suggest_patterns(self, model: dict) -> list:
        """Suggest potential regex patterns based on model data"""
        suggestions = []
        
        # Analyze all sources
        all_text = ' '.join(filter(None, [
            model.get('model_name', ''),
            model.get('source_path', ''),
            model.get('file_name', '')
        ]))
        
        # Extract potential version patterns
        version_patterns = re.findall(r'(?i)\b(sd|sdxl|flux|pony)\s*[._-]*\s*\d+\.?\d*', all_text)
        for pattern in version_patterns:
            suggestions.append(f"Pattern found: {pattern}")
        
        # Extract directory names that might indicate base model
        if model.get('source_path'):
            path_parts = model['source_path'].split('/')
            for part in path_parts:
                if re.search(r'(?i)(sd|sdxl|flux|pony|v\d)', part):
                    suggestions.append(f"Directory indicator: {part}")
        
        return suggestions[:3]  # Limit suggestions
    
    def get_models_to_sort(self) -> List[Dict]:
        """Get all models that need to be sorted - includes both processed and unprocessed models"""
        cursor = self.conn.cursor()
        
        # Get all model files from scanned_files that haven't been successfully moved yet
        cursor.execute('''
            SELECT 
                COALESCE(mf.id, -1) as id,
                sf.file_path as source_path, 
                sf.file_name,
                sf.file_size,
                sf.sha256,
                sf.blake3,
                COALESCE(mf.model_name, NULL) as model_name,
                COALESCE(mf.base_model, NULL) as base_model,
                COALESCE(mf.model_type, NULL) as model_type,
                COALESCE(mf.status, 'unprocessed') as status,
                COALESCE(mf.is_duplicate, 0) as is_duplicate,
                COALESCE(mf.target_path, NULL) as target_path,
                sf.id as scanned_file_id
            FROM scanned_files sf
            LEFT JOIN model_files mf ON sf.id = mf.scanned_file_id
            WHERE sf.file_type = 'model'
            AND (mf.status IS NULL OR mf.status IN ('pending', 'unprocessed'))
            AND (mf.is_duplicate IS NULL OR mf.is_duplicate = 0)
            ORDER BY 
                CASE WHEN mf.model_name IS NULL OR mf.model_name = 'None' THEN 1 ELSE 0 END,
                COALESCE(mf.base_model, 'Unknown'), 
                COALESCE(mf.model_name, sf.file_name)
        ''')
        
        models = [dict(row) for row in cursor.fetchall()]
        
        print(f"📊 Found {len(models)} models to sort (including unprocessed models)")
        
        # Enhance models with civitai information
        print(f"🔍 Enhancing model information using civitai database...")
        enhanced_models = []
        enhanced_count = 0
        
        for model in models:
            enhanced_model = self.enhance_model_info_from_civitai(model)
            if enhanced_model.get('civitai_type'):
                enhanced_count += 1
            enhanced_models.append(enhanced_model)
        
        if enhanced_count > 0:
            print(f"✨ Enhanced {enhanced_count} models with civitai metadata")
        
        return enhanced_models
    
    def get_duplicates_to_move(self) -> List[Dict]:
        """Get all duplicate models that need to be moved"""
        cursor = self.conn.cursor()
        
        cursor.execute('''
            SELECT mf.*, sf.file_path as source_path, sf.file_name,
                   ROW_NUMBER() OVER (PARTITION BY mf.duplicate_group_id ORDER BY sf.created_at) as duplicate_number
            FROM model_files mf
            JOIN scanned_files sf ON mf.scanned_file_id = sf.id
            WHERE mf.is_duplicate = 1 AND mf.status = 'pending'
            ORDER BY mf.duplicate_group_id, sf.created_at
        ''')
        
        return [dict(row) for row in cursor.fetchall()]
    
    def sort_all_models(self):
        """Sort all models into their target directories"""
        print("Starting model sorting...")
        
        # First, move duplicates to duplicates folder
        duplicates = self.get_duplicates_to_move()
        if duplicates:
            print(f"\nMoving {len(duplicates)} duplicate models...")
            for duplicate in duplicates:
                print(f"Processing duplicate: {duplicate['model_name']}")
                self.move_duplicate_model(duplicate)
        
        # Then, move primary models to their sorted locations
        models = self.get_models_to_sort()
        if models:
            # Count models by name status for user info
            named_count = sum(1 for m in models if m['model_name'] and m['model_name'] != 'None')
            unnamed_count = len(models) - named_count
            
            print(f"\nSorting {len(models)} models...")
            print(f"  📋 {named_count} models with proper names (processing first)")
            print(f"  ❓ {unnamed_count} models with missing/null names (processing last)")
            
            for i, model in enumerate(models, 1):
                progress = f"[{i}/{len(models)}]"
                model_name = model['model_name'] or 'None'
                base_model = model['base_model'] or 'Unknown'
                print(f"{progress} Processing model: {model_name} ({base_model})")
                
                # If model info was enhanced, update the database
                if model.get('civitai_type') and model['model_name'] != 'None':
                    cursor = self.conn.cursor()
                    cursor.execute('''
                        UPDATE model_files 
                        SET model_name = ?, base_model = ?, updated_at = strftime('%s', 'now')
                        WHERE id = ?
                    ''', (model['model_name'], model['base_model'], model['id']))
                    self.conn.commit()
                
                self.move_model_and_associated_files(model)
        
        print(f"\n{'='*50}")
        print("SORTING COMPLETE")
        print(f"{'='*50}")
        print(f"Models moved: {self.stats['models_moved']}")
        print(f"Duplicates moved: {self.stats['duplicates_moved']}")
        print(f"Associated files moved: {self.stats['associated_files_moved']}")
        print(f"Errors: {self.stats['errors']}")
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Sort and organize model files')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without actually moving files')
    parser.add_argument('--config', default='config.ini', help='Configuration file path')
    
    args = parser.parse_args()
    
    sorter = ModelSorter(args.config, dry_run=args.dry_run)
    
    try:
        if args.dry_run:
            print("DRY RUN MODE - No files will be moved")
            print("="*50)
        
        sorter.sort_all_models()
        
    except Exception as e:
        print(f"Error during sorting: {e}")
    finally:
        sorter.close()


if __name__ == "__main__":
    main()