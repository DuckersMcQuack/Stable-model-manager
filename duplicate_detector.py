#!/usr/bin/env python3
"""
Duplicate Detection System
Compares hashes and metadata to identify duplicates with intelligence for metadata preservation
"""

import json
import sqlite3
from typing import Dict, List, Tuple, Optional, Set
import os
import time


class DuplicateDetector:
    """Handles duplicate detection and management"""
    
    def __init__(self, db_path: str = "model_sorter.sqlite"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
    
    def find_duplicate_hashes(self) -> Dict[str, List[Dict]]:
        """Find all duplicate files by SHA256 hash"""
        cursor = self.conn.cursor()
        
        # Find hashes that appear multiple times
        cursor.execute('''
            SELECT sha256, COUNT(*) as count
            FROM scanned_files 
            WHERE file_type = 'model'
            GROUP BY sha256 
            HAVING COUNT(*) > 1
        ''')
        
        duplicate_hashes = {}
        
        for row in cursor.fetchall():
            sha256 = row['sha256']
            count = row['count']
            
            # Get all files with this hash
            cursor.execute('''
                SELECT sf.*, mf.model_name, mf.base_model, mf.model_type, 
                       mf.has_civitai_info, mf.has_metadata_json, mf.metadata_json,
                       mf.id as model_id
                FROM scanned_files sf
                LEFT JOIN model_files mf ON sf.id = mf.scanned_file_id
                WHERE sf.sha256 = ? AND sf.file_type = 'model'
                ORDER BY sf.created_at
            ''', (sha256,))
            
            files = [dict(row) for row in cursor.fetchall()]
            duplicate_hashes[sha256] = files
        
        return duplicate_hashes
    
    def find_autov3_duplicate_hashes(self) -> Dict[str, List[Dict]]:
        """Find all duplicate files by AutoV3 hash (SafeTensors model weights only)"""
        cursor = self.conn.cursor()
        
        # Find AutoV3 hashes that appear multiple times
        cursor.execute('''
            SELECT autov3, COUNT(*) as count
            FROM scanned_files 
            WHERE file_type = 'model' AND autov3 IS NOT NULL
            GROUP BY autov3 
            HAVING COUNT(*) > 1
        ''')
        
        duplicate_hashes = {}
        
        for row in cursor.fetchall():
            autov3 = row['autov3']
            count = row['count']
            
            # Get all files with this AutoV3 hash
            cursor.execute('''
                SELECT sf.*, mf.model_name, mf.base_model, mf.model_type, 
                       mf.has_civitai_info, mf.has_metadata_json, mf.metadata_json,
                       mf.id as model_id
                FROM scanned_files sf
                LEFT JOIN model_files mf ON sf.id = mf.scanned_file_id
                WHERE sf.autov3 = ? AND sf.file_type = 'model'
                ORDER BY sf.created_at
            ''', (autov3,))
            
            files = [dict(row) for row in cursor.fetchall()]
            duplicate_hashes[autov3] = files
        
        return duplicate_hashes
    
    def score_file_completeness(self, file_info: Dict) -> int:
        """Score a file based on how complete its metadata and associated files are"""
        score = 0
        
        # Base score for existence
        score += 1
        
        # Bonus for having civitai.info
        if file_info.get('has_civitai_info'):
            score += 10
        
        # Bonus for having metadata.json
        if file_info.get('has_metadata_json'):
            score += 5
        
        # Parse metadata to check for additional info
        if file_info.get('metadata_json'):
            try:
                metadata = json.loads(file_info['metadata_json'])
                
                # Bonus for having trained words
                if metadata.get('trained_words'):
                    score += 3
                
                # Bonus for having civitai data
                if metadata.get('civitai_data'):
                    score += 5
                
                # Bonus for having associated files
                if metadata.get('associated_files'):
                    assoc_files = metadata['associated_files']
                    if assoc_files.get('civitai_info'):
                        score += 2
                    if assoc_files.get('metadata_json'):
                        score += 2
                    if assoc_files.get('other_text_files'):
                        score += len(assoc_files['other_text_files'])
            except:
                pass
        
        # Check for associated image files
        if file_info.get('file_path'):
            model_dir = os.path.dirname(file_info['file_path'])
            model_name = os.path.splitext(os.path.basename(file_info['file_path']))[0]
            
            # Count associated images
            image_extensions = ['.png', '.jpg', '.jpeg', '.webp', '.gif']
            for ext in image_extensions:
                image_path = os.path.join(model_dir, f"{model_name}.preview{ext}")
                if os.path.exists(image_path):
                    score += 2
                
                image_path = os.path.join(model_dir, f"{model_name}{ext}")
                if os.path.exists(image_path):
                    score += 2
        
        return score
    
    def determine_primary_duplicate(self, duplicate_files: List[Dict]) -> Tuple[Optional[Dict], List[Dict]]:
        """Determine which file should be the primary (kept) version"""
        if len(duplicate_files) <= 1:
            return duplicate_files[0] if duplicate_files else None, []
        
        # Score each file
        scored_files = []
        for file_info in duplicate_files:
            score = self.score_file_completeness(file_info)
            scored_files.append((score, file_info))
        
        # Sort by score (highest first), then by creation time (earliest first)
        scored_files.sort(key=lambda x: (-x[0], x[1].get('created_at', 0)))
        
        primary_file = scored_files[0][1]
        duplicate_files = [f[1] for f in scored_files[1:]]
        
        return primary_file, duplicate_files
    
    def merge_metadata_from_duplicates(self, primary_file: Dict, duplicates: List[Dict]) -> Dict:
        """Merge metadata from duplicate files into the primary file"""
        primary_metadata = {}
        
        # Start with primary file metadata
        if primary_file.get('metadata_json'):
            try:
                primary_metadata = json.loads(primary_file['metadata_json'])
            except:
                pass
        
        merged_metadata = primary_metadata.copy()
        
        # Merge metadata from duplicates
        for dup_file in duplicates:
            if dup_file.get('metadata_json'):
                try:
                    dup_metadata = json.loads(dup_file['metadata_json'])
                    
                    # Merge trained words
                    primary_words = set(merged_metadata.get('trained_words', []))
                    dup_words = set(dup_metadata.get('trained_words', []))
                    merged_metadata['trained_words'] = list(primary_words | dup_words)
                    
                    # If primary doesn't have civitai data but duplicate does, use it
                    if not merged_metadata.get('civitai_data') and dup_metadata.get('civitai_data'):
                        merged_metadata['civitai_data'] = dup_metadata['civitai_data']
                    
                    # If primary doesn't have metadata_json data but duplicate does, use it
                    if not merged_metadata.get('metadata_json_data') and dup_metadata.get('metadata_json_data'):
                        merged_metadata['metadata_json_data'] = dup_metadata['metadata_json_data']
                    
                    # Merge associated files
                    if not merged_metadata.get('associated_files') and dup_metadata.get('associated_files'):
                        merged_metadata['associated_files'] = dup_metadata['associated_files']
                    elif merged_metadata.get('associated_files') and dup_metadata.get('associated_files'):
                        # Merge associated files intelligently
                        merged_assoc = merged_metadata['associated_files']
                        dup_assoc = dup_metadata['associated_files']
                        
                        # Use best civitai.info
                        if not merged_assoc.get('civitai_info') and dup_assoc.get('civitai_info'):
                            merged_assoc['civitai_info'] = dup_assoc['civitai_info']
                        
                        # Use best metadata.json
                        if not merged_assoc.get('metadata_json') and dup_assoc.get('metadata_json'):
                            merged_assoc['metadata_json'] = dup_assoc['metadata_json']
                        
                        # Combine other text files
                        merged_others = merged_assoc.get('other_text_files', [])
                        dup_others = dup_assoc.get('other_text_files', [])
                        
                        # Deduplicate by file path
                        existing_paths = {f['path'] for f in merged_others}
                        for dup_file in dup_others:
                            if dup_file['path'] not in existing_paths:
                                merged_others.append(dup_file)
                        
                        merged_assoc['other_text_files'] = merged_others
                        merged_metadata['associated_files'] = merged_assoc
                
                except Exception as e:
                    print(f"Error merging metadata: {e}")
                    continue
        
        return merged_metadata
    
    def create_duplicate_groups(self) -> Dict[str, Dict]:
        """Create duplicate groups and determine primary files"""
        # Find duplicates by SHA256 (exact file matches)
        sha256_duplicates = self.find_duplicate_hashes()
        
        # Find duplicates by AutoV3 (same model weights, different metadata)
        autov3_duplicates = self.find_autov3_duplicate_hashes()
        
        duplicate_groups = {}
        cursor = self.conn.cursor()
        
        # Process SHA256 duplicates first (exact matches)
        for sha256, duplicate_files in sha256_duplicates.items():
            print(f"\nProcessing duplicates for hash {sha256[:16]}...")
            print(f"Found {len(duplicate_files)} duplicate files:")
            
            for i, file_info in enumerate(duplicate_files):
                print(f"  {i+1}. {file_info['file_path']}")
                print(f"     Score: {self.score_file_completeness(file_info)}")
                print(f"     Has civitai.info: {file_info.get('has_civitai_info', False)}")
                print(f"     Has metadata.json: {file_info.get('has_metadata_json', False)}")
            
            # Determine primary file
            primary_file, duplicates = self.determine_primary_duplicate(duplicate_files)
            
            if primary_file is None:
                print(f"\nNo primary file found for group {sha256}, skipping...")
                continue
            
            print(f"\nPrimary file: {primary_file['file_path']}")
            print(f"Duplicates to move: {len(duplicates)}")
            
            # Merge metadata
            merged_metadata = self.merge_metadata_from_duplicates(primary_file, duplicates)
            
            # Create duplicate group record
            cursor.execute('''
                INSERT OR REPLACE INTO duplicate_groups (sha256, primary_model_id, duplicate_count)
                VALUES (?, ?, ?)
            ''', (sha256, primary_file.get('model_id'), len(duplicate_files)))
            
            group_id = cursor.lastrowid
            
            # Update model records to mark duplicates
            for i, file_info in enumerate(duplicate_files):
                is_primary = (file_info['id'] == primary_file['id'])
                
                cursor.execute('''
                    UPDATE model_files 
                    SET is_duplicate = ?, duplicate_group_id = ?, updated_at = ?
                    WHERE scanned_file_id = ?
                ''', (not is_primary, group_id, int(time.time()), file_info['id']))
            
            # Update primary file with merged metadata
            cursor.execute('''
                UPDATE model_files 
                SET metadata_json = ?, updated_at = ?
                WHERE scanned_file_id = ?
            ''', (json.dumps(merged_metadata), int(time.time()), primary_file['id']))
            
            duplicate_groups[sha256] = {
                'group_id': group_id,
                'primary_file': primary_file,
                'duplicates': duplicates,
                'merged_metadata': merged_metadata
            }
        
        # Process AutoV3 duplicates (same model weights, different metadata)
        # Only process AutoV3 duplicates that weren't already handled as SHA256 duplicates
        processed_file_ids = set()
        for group_data in duplicate_groups.values():
            processed_file_ids.add(group_data['primary_file']['id'])
            for dup in group_data['duplicates']:
                processed_file_ids.add(dup['id'])
        
        for autov3, duplicate_files in autov3_duplicates.items():
            # Skip if all files in this AutoV3 group were already processed as SHA256 duplicates
            unprocessed_files = [f for f in duplicate_files if f['id'] not in processed_file_ids]
            if len(unprocessed_files) < 2:
                continue
                
            print(f"\nProcessing AutoV3 duplicates for hash {autov3[:16]}...")
            print(f"Found {len(unprocessed_files)} AutoV3 duplicate files (same model weights, different metadata):")
            
            for i, file_info in enumerate(unprocessed_files):
                print(f"  {i+1}. {file_info['file_path']}")
                print(f"     Score: {self.score_file_completeness(file_info)}")
                print(f"     Has civitai.info: {file_info.get('has_civitai_info', False)}")
                print(f"     Has metadata.json: {file_info.get('has_metadata_json', False)}")
            
            # Determine primary file
            primary_file, duplicates = self.determine_primary_duplicate(unprocessed_files)
            
            if primary_file is None:
                print(f"\nNo primary file found for AutoV3 group {autov3}, skipping...")
                continue
            
            print(f"\nPrimary file: {primary_file['file_path']}")
            print(f"Duplicates to move: {len(duplicates)}")
            
            # Merge metadata
            merged_metadata = self.merge_metadata_from_duplicates(primary_file, duplicates)
            
            # Create duplicate group in database
            group_id = abs(hash(autov3)) % 1000000000
            cursor.execute('''
                INSERT OR REPLACE INTO duplicate_groups 
                (id, sha256, primary_model_id, duplicate_count, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (group_id, autov3, primary_file.get('model_id'), len(duplicates), 
                  int(time.time()), int(time.time())))
            
            # Update model records to mark duplicates
            for i, file_info in enumerate(unprocessed_files):
                is_primary = (file_info['id'] == primary_file['id'])
                
                cursor.execute('''
                    UPDATE model_files 
                    SET is_duplicate = ?, duplicate_group_id = ?, updated_at = ?
                    WHERE scanned_file_id = ?
                ''', (not is_primary, group_id, int(time.time()), file_info['id']))
                
                # Add to processed set
                processed_file_ids.add(file_info['id'])
            
            # Update primary file with merged metadata
            cursor.execute('''
                UPDATE model_files 
                SET metadata_json = ?, updated_at = ?
                WHERE scanned_file_id = ?
            ''', (json.dumps(merged_metadata), int(time.time()), primary_file['id']))
            
            duplicate_groups[f"autov3_{autov3}"] = {
                'group_id': group_id,
                'primary_file': primary_file,
                'duplicates': duplicates,
                'merged_metadata': merged_metadata,
                'type': 'autov3'  # Indicate this is an AutoV3 duplicate group
            }

        self.conn.commit()
        return duplicate_groups
    
    def get_files_to_move_to_duplicates_folder(self) -> List[Dict]:
        """Get list of files that should be moved to duplicates folder"""
        cursor = self.conn.cursor()
        
        cursor.execute('''
            SELECT sf.*, mf.model_name, mf.base_model, mf.duplicate_group_id,
                   dg.sha256 as group_sha256
            FROM scanned_files sf
            JOIN model_files mf ON sf.id = mf.scanned_file_id
            JOIN duplicate_groups dg ON mf.duplicate_group_id = dg.id
            WHERE mf.is_duplicate = 1 AND sf.file_type = 'model'
            ORDER BY mf.duplicate_group_id, sf.created_at
        ''')
        
        duplicate_files = []
        group_counters = {}
        
        for row in cursor.fetchall():
            file_info = dict(row)
            group_id = file_info['duplicate_group_id']
            
            # Generate numbered folder name for duplicates
            if group_id not in group_counters:
                group_counters[group_id] = 1
            else:
                group_counters[group_id] += 1
            
            file_info['duplicate_number'] = group_counters[group_id]
            duplicate_files.append(file_info)
        
        return duplicate_files
    
    def plan_duplicate_moves(self) -> Dict:
        """Plan all duplicate file moves"""
        duplicate_files = self.get_files_to_move_to_duplicates_folder()
        
        move_plan = {
            'duplicates_to_move': [],
            'metadata_to_migrate': [],
            'summary': {
                'total_duplicates': len(duplicate_files),
                'groups_affected': len(set(f['duplicate_group_id'] for f in duplicate_files))
            }
        }
        
        for file_info in duplicate_files:
            model_name = file_info['model_name'] or os.path.splitext(file_info['file_name'])[0]
            dup_number = file_info['duplicate_number']
            
            # Plan duplicate folder structure
            duplicate_folder_name = f"{model_name}_{dup_number}" if dup_number > 1 else f"{model_name}_1"
            
            move_info = {
                'source_path': file_info['file_path'],
                'target_folder': f"loras/duplicates/{duplicate_folder_name}",
                'target_path': f"loras/duplicates/{duplicate_folder_name}/{file_info['file_name']}",
                'model_name': model_name,
                'duplicate_number': dup_number,
                'group_id': file_info['duplicate_group_id']
            }
            
            move_plan['duplicates_to_move'].append(move_info)
        
        return move_plan
    
    def get_duplicate_summary(self) -> Dict:
        """Get summary of duplicate detection results"""
        cursor = self.conn.cursor()
        
        # Count duplicate groups
        cursor.execute('SELECT COUNT(*) FROM duplicate_groups')
        group_count = cursor.fetchone()[0]
        
        # Count duplicate files
        cursor.execute('SELECT COUNT(*) FROM model_files WHERE is_duplicate = 1')
        duplicate_count = cursor.fetchone()[0]
        
        # Get group details
        cursor.execute('''
            SELECT dg.sha256, dg.duplicate_count, 
                   COUNT(mf.id) as files_in_group
            FROM duplicate_groups dg
            LEFT JOIN model_files mf ON dg.id = mf.duplicate_group_id
            GROUP BY dg.id
        ''')
        
        group_details = [dict(row) for row in cursor.fetchall()]
        
        return {
            'duplicate_groups': group_count,
            'duplicate_files': duplicate_count,
            'group_details': group_details
        }
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


def main():
    """Main function"""
    import time
    
    detector = DuplicateDetector()
    
    try:
        print("Starting duplicate detection...")
        
        # Create duplicate groups
        duplicate_groups = detector.create_duplicate_groups()
        
        print(f"\n{'='*50}")
        print("DUPLICATE DETECTION COMPLETE")
        print(f"{'='*50}")
        
        # Show summary
        summary = detector.get_duplicate_summary()
        print(f"Duplicate groups found: {summary['duplicate_groups']}")
        print(f"Duplicate files found: {summary['duplicate_files']}")
        
        if duplicate_groups:
            print(f"\nDuplicate groups:")
            for sha256, group_info in duplicate_groups.items():
                print(f"  Hash {sha256[:16]}:")
                print(f"    Primary: {group_info['primary_file']['file_path']}")
                print(f"    Duplicates: {len(group_info['duplicates'])}")
        
        # Plan moves
        move_plan = detector.plan_duplicate_moves()
        print(f"\nMove plan:")
        print(f"  Files to move to duplicates folder: {len(move_plan['duplicates_to_move'])}")
        print(f"  Groups affected: {move_plan['summary']['groups_affected']}")
        
        if move_plan['duplicates_to_move']:
            print(f"\nPlanned moves:")
            for move in move_plan['duplicates_to_move'][:3]:  # Show first 3
                print(f"  {move['source_path']} -> {move['target_path']}")
        
    except Exception as e:
        print(f"Error during duplicate detection: {e}")
    finally:
        detector.close()


if __name__ == "__main__":
    main()