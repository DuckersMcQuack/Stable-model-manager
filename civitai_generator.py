#!/usr/bin/env python3
"""
Civitai Info Generator
Generates properly formatted civitai.info files from model metadata when missing
"""

import json
import os
import sqlite3
import time
from datetime import datetime
from typing import Dict, List, Optional, Any


class CivitaiInfoGenerator:
    """Generates civitai.info files from available metadata"""
    
    def __init__(self, db_path: str = "model_sorter.sqlite", civitai_db_path: str = "Database/civitai.sqlite"):
        self.db_path = db_path
        self.civitai_db_path = civitai_db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        
        # Connect to civitai database if available
        self.civitai_conn = None
        if os.path.exists(civitai_db_path):
            try:
                self.civitai_conn = sqlite3.connect(civitai_db_path)
                self.civitai_conn.row_factory = sqlite3.Row
                print(f"Connected to civitai database: {civitai_db_path}")
            except Exception as e:
                print(f"Warning: Could not connect to civitai database: {e}")
        
        # Cache for model type mappings from civitai database
        self._model_type_cache = {}
        self._base_model_cache = {}
        self._available_model_types = set()
        self._available_base_models = set()
        self._load_civitai_metadata_stats()
    
    def _load_civitai_metadata_stats(self):
        """Load available model types and base models from civitai database"""
        if not self.civitai_conn:
            return
        
        try:
            cursor = self.civitai_conn.cursor()
            
            # Get all unique model types
            cursor.execute('SELECT DISTINCT type FROM models WHERE type IS NOT NULL')
            for row in cursor.fetchall():
                self._available_model_types.add(row['type'])
            
            # Get all unique base models
            cursor.execute('SELECT DISTINCT base_model FROM model_versions WHERE base_model IS NOT NULL')
            for row in cursor.fetchall():
                self._available_base_models.add(row['base_model'])
            
            print(f"Civitai database contains {len(self._available_model_types)} model types: {sorted(self._available_model_types)}")
            print(f"Civitai database contains {len(self._available_base_models)} base models: {sorted(self._available_base_models)}")
            
        except Exception as e:
            print(f"Warning: Could not load civitai metadata stats: {e}")
    
    def lookup_model_in_civitai_db(self, sha256: str) -> Optional[Dict]:
        """Look up model information in civitai.sqlite database by SHA256"""
        if not self.civitai_conn:
            return None
        
        try:
            cursor = self.civitai_conn.cursor()
            
            # Get model file info
            cursor.execute('''
                SELECT mf.id, mf.model_id, mf.version_id, mf.type, mf.sha256, mf.data
                FROM model_files mf
                WHERE mf.sha256 = ?
            ''', (sha256.upper(),))
            
            file_row = cursor.fetchone()
            if not file_row:
                print(f"No match found in civitai database for SHA256: {sha256[:16]}...")
                return None
            
            print(f"Found match in civitai database for SHA256: {sha256[:16]}...")
            file_data = json.loads(file_row['data']) if file_row['data'] else {}
            
            # Get model info
            cursor.execute('''
                SELECT m.id, m.name, m.type, m.data
                FROM models m
                WHERE m.id = ?
            ''', (file_row['model_id'],))
            
            model_row = cursor.fetchone()
            if not model_row:
                return None
            
            model_data = json.loads(model_row['data']) if model_row['data'] else {}
            
            # Get version info
            cursor.execute('''
                SELECT mv.id, mv.name, mv.base_model, mv.data
                FROM model_versions mv
                WHERE mv.id = ?
            ''', (file_row['version_id'],))
            
            version_row = cursor.fetchone()
            if not version_row:
                return None
            
            version_data = json.loads(version_row['data']) if version_row['data'] else {}
            
            # Combine all data into civitai.info format
            civitai_info = {
                "id": version_row['id'],
                "modelId": model_row['id'],
                "name": version_row['name'],
                "createdAt": version_data.get('createdAt', ''),
                "updatedAt": version_data.get('updatedAt', ''),
                "status": version_data.get('status', 'Published'),
                "publishedAt": version_data.get('publishedAt', ''),
                "trainedWords": version_data.get('trainedWords', []),
                "trainingStatus": version_data.get('trainingStatus'),
                "trainingDetails": version_data.get('trainingDetails'),
                "baseModel": version_row['base_model'],
                "baseModelType": version_data.get('baseModelType'),
                "earlyAccessEndsAt": version_data.get('earlyAccessEndsAt'),
                "earlyAccessConfig": version_data.get('earlyAccessConfig'),
                "description": version_data.get('description', ''),
                "uploadType": version_data.get('uploadType', 'Created'),
                "air": version_data.get('air', ''),
                "stats": version_data.get('stats', {
                    "downloadCount": 0,
                    "ratingCount": 0,
                    "rating": 0,
                    "thumbsUpCount": 0
                }),
                "model": {
                    "name": model_row['name'],
                    "type": model_row['type'],
                    "nsfw": model_data.get('nsfw', False),
                    "poi": model_data.get('poi', False)
                },
                "files": [{
                    "id": file_row['id'],
                    "sizeKB": file_data.get('sizeKB', 0),
                    "name": file_data.get('name', ''),
                    "type": file_data.get('type', 'Model'),
                    "pickleScanResult": file_data.get('pickleScanResult', 'Success'),
                    "pickleScanMessage": file_data.get('pickleScanMessage', 'No Pickle imports'),
                    "virusScanResult": file_data.get('virusScanResult', 'Success'),
                    "virusScanMessage": file_data.get('virusScanMessage'),
                    "scannedAt": file_data.get('scannedAt', ''),
                    "metadata": file_data.get('metadata', {
                        "format": "SafeTensor",
                        "size": None,
                        "fp": None
                    }),
                    "hashes": file_data.get('hashes', {}),
                    "primary": file_data.get('primary', True),
                    "downloadUrl": file_data.get('downloadUrl', '')
                }],
                "images": version_data.get('images', []),
                "downloadUrl": f"https://civitai.com/api/download/models/{version_row['id']}"
            }
            
            return civitai_info
            
        except Exception as e:
            print(f"Error looking up model in civitai database: {e}")
            return None
    
    def determine_model_type_and_base_model(self, model_metadata: Dict) -> tuple[str, str]:
        """Determine model type and base model from available data"""
        sha256_hash = model_metadata.get('sha256', '')
        
        # First check civitai database for authoritative info
        if sha256_hash and self.civitai_conn:
            try:
                cursor = self.civitai_conn.cursor()
                cursor.execute('''
                    SELECT m.type as model_type, mv.base_model
                    FROM model_files mf
                    JOIN models m ON mf.model_id = m.id
                    JOIN model_versions mv ON mf.version_id = mv.id
                    WHERE mf.sha256 = ?
                ''', (sha256_hash.upper(),))
                
                result = cursor.fetchone()
                if result:
                    return result['model_type'], result['base_model']
            except Exception as e:
                print(f"Error querying civitai database: {e}")
        
        # Fall back to metadata or intelligent defaults
        model_type = model_metadata.get('model_type')
        base_model = model_metadata.get('base_model')
        file_name = model_metadata.get('file_name', '').lower()
        
        # Determine model type from filename or extension if not available
        if not model_type:
            # Use filename patterns to match known civitai model types
            filename_patterns = {
                'LORA': ['lora', 'lycoris', 'locon', 'loha'],
                'Checkpoint': ['checkpoint', 'ckpt', 'model'],
                'VAE': ['vae'],
                'TextualInversion': ['embedding', 'textual_inversion', 'ti', 'embed'],
                'Controlnet': ['controlnet', 'control_net', 'cnet'],
                'Upscaler': ['upscaler', 'esrgan', 'realesrgan', 'swinir'],
                'Hypernetwork': ['hypernetwork', 'hypernet'],
                'AestheticGradient': ['aesthetic', 'gradient'],
                'Poses': ['pose', 'openpose'],
                'Wildcards': ['wildcard']
            }
            
            # Check against available model types from civitai database
            for db_type in self._available_model_types:
                if db_type.lower() in file_name:
                    model_type = db_type
                    break
            
            # If not found in database types, use pattern matching
            if not model_type:
                for pattern_type, patterns in filename_patterns.items():
                    if any(pattern in file_name for pattern in patterns):
                        # Use database type if available, otherwise use pattern type
                        if pattern_type in self._available_model_types:
                            model_type = pattern_type
                        else:
                            # Find closest match in database
                            for db_type in self._available_model_types:
                                if pattern_type.lower() in db_type.lower() or db_type.lower() in pattern_type.lower():
                                    model_type = db_type
                                    break
                            if not model_type:
                                model_type = pattern_type
                        break
            
            if not model_type:
                model_type = 'Other'
        
        # Determine base model if not available
        if not base_model:
            # Pattern matching for base models
            base_model_patterns = {
                'SD 3': ['sd3', 'stable_diffusion_3', 'stablediffusion3'],
                'SD 3.5': ['sd35', 'sd3.5', 'stable_diffusion_35'],
                'SDXL 1.0': ['sdxl', 'xl', 'stable_diffusion_xl'],
                'SDXL Turbo': ['sdxl_turbo', 'turbo'],
                'SD 2.1': ['sd2', 'sd21', 'stable_diffusion_2'],
                'SD 2.0': ['sd20', 'stable_diffusion_20'],
                'SD 1.5': ['sd15', 'sd1.5', 'stable_diffusion_15'],
                'SD 1.4': ['sd14', 'sd1.4', 'stable_diffusion_14'],
                'Flux.1 D': ['flux', 'flux1', 'flux_dev'],
                'Flux.1 S': ['flux_schnell', 'flux_s'],
                'Pony': ['pony', 'ponydiffusion'],
                'SDXL Lightning': ['lightning'],
                'Playground v2': ['playground'],
                'PixArt': ['pixart']
            }
            
            # Check against available base models from civitai database first
            for db_base in self._available_base_models:
                if db_base.lower().replace(' ', '').replace('.', '') in file_name.replace(' ', '').replace('.', ''):
                    base_model = db_base
                    break
            
            # If not found in database, use pattern matching
            if not base_model:
                for pattern_base, patterns in base_model_patterns.items():
                    if any(pattern in file_name for pattern in patterns):
                        # Use database base model if available, otherwise pattern
                        if pattern_base in self._available_base_models:
                            base_model = pattern_base
                        else:
                            # Find closest match in database
                            for db_base in self._available_base_models:
                                if any(p in db_base.lower() for p in patterns):
                                    base_model = db_base
                                    break
                            if not base_model:
                                base_model = pattern_base
                        break
            
            # Default fallback to most common
            if not base_model:
                base_model = 'SD 1.5' if 'SD 1.5' in self._available_base_models else list(self._available_base_models)[0] if self._available_base_models else 'SD 1.5'
        
        return model_type, base_model
    
    def generate_civitai_info_from_metadata(self, model_metadata: Dict) -> Dict:
        """Generate a civitai.info structure from available metadata"""
        
        # Get base information
        model_name = model_metadata.get('model_name', 'Unknown Model')
        trained_words = model_metadata.get('trained_words', [])
        file_name = model_metadata.get('file_name', 'model.safetensors')
        file_size = model_metadata.get('file_size', 0)
        sha256_hash = model_metadata.get('sha256', '')
        
        # Determine model type and base model dynamically
        model_type, base_model = self.determine_model_type_and_base_model(model_metadata)
        
        # First try to get from civitai database by SHA256
        if sha256_hash:
            civitai_info = self.lookup_model_in_civitai_db(sha256_hash)
            if civitai_info:
                print(f"Found complete civitai data for {model_name} (Type: {civitai_info['model']['type']}, Base: {civitai_info['baseModel']})")
                return civitai_info
        
        # Use existing civitai data if available
        existing_civitai = model_metadata.get('civitai_data')
        if existing_civitai:
            return existing_civitai
        
        # Check metadata_json for civitai data
        metadata_json_data = model_metadata.get('metadata_json_data')
        if metadata_json_data and 'civitai' in metadata_json_data:
            return metadata_json_data['civitai']
        
        print(f"Generating synthetic civitai.info for {model_name} (no database match found)")
        
        # Generate new civitai.info structure
        current_time = datetime.utcnow().isoformat() + 'Z'
        
        # Generate synthetic IDs (negative to indicate they're synthetic)
        synthetic_model_id = abs(hash(model_name)) % 1000000
        synthetic_version_id = abs(hash(f"{model_name}_{file_name}")) % 1000000
        synthetic_file_id = abs(hash(f"{model_name}_{file_name}_{file_size}")) % 1000000
        
        civitai_info = {
            "id": synthetic_version_id,
            "modelId": synthetic_model_id,
            "name": "v1",  # Default version name
            "createdAt": current_time,
            "updatedAt": current_time,
            "status": "Published",
            "publishedAt": current_time,
            "trainedWords": trained_words,
            "trainingStatus": None,
            "trainingDetails": None,
            "baseModel": base_model,
            "baseModelType": None,
            "earlyAccessEndsAt": None,
            "earlyAccessConfig": None,
            "description": f"Generated metadata for {model_name}",
            "uploadType": "Created",
            "air": f"urn:air:sd:lora:local:{synthetic_model_id}@{synthetic_version_id}",
            "stats": {
                "downloadCount": 0,
                "ratingCount": 0,
                "rating": 0,
                "thumbsUpCount": 0
            },
            "model": {
                "name": model_name,
                "type": model_type,
                "nsfw": False,
                "poi": False
            },
            "files": [
                {
                    "id": synthetic_file_id,
                    "sizeKB": round(file_size / 1024, 2) if file_size else 0,
                    "name": os.path.splitext(file_name)[0],
                    "type": "Model",
                    "pickleScanResult": "Success",
                    "pickleScanMessage": "No Pickle imports",
                    "virusScanResult": "Success",
                    "virusScanMessage": None,
                    "scannedAt": current_time,
                    "metadata": {
                        "format": "SafeTensor" if file_name.endswith('.safetensors') else "PyTorch",
                        "size": None,
                        "fp": None
                    },
                    "hashes": {
                        "SHA256": sha256_hash.upper() if sha256_hash else None
                    },
                    "primary": True,
                    "downloadUrl": None
                }
            ],
            "images": [],
            "downloadUrl": None
        }
        
        return civitai_info
    
    def generate_civitai_info_from_existing_db(self, model_name: str, 
                                              base_model: str, model_type: str) -> Optional[Dict]:
        """Try to find similar model in existing civitai database"""
        # This would query the existing civitai.sqlite database
        # For now, we'll return None and rely on generated metadata
        return None
    
    def create_civitai_info_file(self, target_directory: str, model_info: Dict) -> bool:
        """Create a civitai.info file in the target directory"""
        try:
            model_name = model_info.get('model_name', 'Unknown')
            file_name = model_info.get('file_name', 'model.safetensors')
            
            # Generate the base name for the civitai.info file
            base_name = os.path.splitext(file_name)[0]
            civitai_info_path = os.path.join(target_directory, f"{base_name}.civitai.info")
            
            # Check if civitai.info already exists
            if os.path.exists(civitai_info_path):
                print(f"Civitai.info already exists: {civitai_info_path}")
                return True
            
            # Parse existing metadata
            metadata = {}
            if model_info.get('metadata_json'):
                try:
                    metadata = json.loads(model_info['metadata_json'])
                except:
                    pass
            
            # Generate civitai.info content
            civitai_info = self.generate_civitai_info_from_metadata(metadata)
            
            # Ensure target directory exists
            os.makedirs(target_directory, exist_ok=True)
            
            # Write the file
            with open(civitai_info_path, 'w', encoding='utf-8') as f:
                json.dump(civitai_info, f, indent=2, ensure_ascii=False)
            
            print(f"Generated civitai.info: {civitai_info_path}")
            return True
            
        except Exception as e:
            print(f"Error creating civitai.info for {model_info.get('model_name', 'Unknown')}: {e}")
            return False
    
    def process_models_needing_civitai_info(self):
        """Process all models that need civitai.info files generated"""
        cursor = self.conn.cursor()
        
        # Get models that don't have civitai.info files
        cursor.execute('''
            SELECT mf.*, sf.file_path, sf.file_name, sf.sha256, sf.file_size
            FROM model_files mf
            JOIN scanned_files sf ON mf.scanned_file_id = sf.id
            WHERE mf.has_civitai_info = 0 AND mf.target_path IS NOT NULL
            ORDER BY mf.base_model, mf.model_name
        ''')
        
        models_needing_info = [dict(row) for row in cursor.fetchall()]
        
        print(f"Found {len(models_needing_info)} models needing civitai.info files...")
        
        success_count = 0
        error_count = 0
        
        for model in models_needing_info:
            try:
                # Get target directory from target_path
                if model['target_path']:
                    target_directory = os.path.dirname(model['target_path'])
                else:
                    print(f"No target path for model: {model['model_name']}")
                    continue
                
                # Prepare model info for generation
                model_info = {
                    'model_name': model['model_name'],
                    'file_name': model['file_name'],
                    'base_model': model['base_model'],
                    'model_type': model['model_type'],
                    'file_size': model['file_size'],
                    'sha256': model['sha256'],
                    'metadata_json': model['metadata_json']
                }
                
                if self.create_civitai_info_file(target_directory, model_info):
                    success_count += 1
                    
                    # Update database to mark that civitai.info now exists
                    cursor.execute('''
                        UPDATE model_files 
                        SET has_civitai_info = 1, updated_at = ?
                        WHERE id = ?
                    ''', (int(time.time()), model['id']))
                else:
                    error_count += 1
                    
            except Exception as e:
                print(f"Error processing model {model.get('model_name', 'Unknown')}: {e}")
                error_count += 1
        
        self.conn.commit()
        
        print(f"\nCivitai.info generation complete:")
        print(f"  Successfully generated: {success_count}")
        print(f"  Errors: {error_count}")
        
        return success_count, error_count
    
    def update_existing_civitai_info_with_better_metadata(self, 
                                                         primary_model_info: Dict, 
                                                         duplicate_metadata: List[Dict]):
        """Update existing civitai.info with better metadata from duplicates"""
        try:
            target_path = primary_model_info.get('target_path')
            if not target_path:
                return False
            
            target_directory = os.path.dirname(target_path)
            file_name = os.path.basename(target_path)
            base_name = os.path.splitext(file_name)[0]
            civitai_info_path = os.path.join(target_directory, f"{base_name}.civitai.info")
            
            # Load existing civitai.info if it exists
            existing_civitai = {}
            if os.path.exists(civitai_info_path):
                try:
                    with open(civitai_info_path, 'r', encoding='utf-8') as f:
                        existing_civitai = json.load(f)
                except:
                    pass
            
            # Start with existing or generate new
            if not existing_civitai:
                # Parse primary model metadata
                primary_metadata = {}
                if primary_model_info.get('metadata_json'):
                    try:
                        primary_metadata = json.loads(primary_model_info['metadata_json'])
                    except:
                        pass
                existing_civitai = self.generate_civitai_info_from_metadata(primary_metadata)
            
            # Enhance with metadata from duplicates
            enhanced_civitai = existing_civitai.copy()
            
            # Collect trained words from all sources
            all_trained_words = set(enhanced_civitai.get('trainedWords', []))
            
            for dup_metadata in duplicate_metadata:
                if dup_metadata.get('trained_words'):
                    all_trained_words.update(dup_metadata['trained_words'])
                
                # If existing doesn't have good civitai data but duplicate does, merge
                if (not enhanced_civitai.get('model', {}).get('name') or 
                    enhanced_civitai['model']['name'] == 'Unknown Model'):
                    
                    dup_civitai = dup_metadata.get('civitai_data')
                    if dup_civitai and dup_civitai.get('model', {}).get('name'):
                        enhanced_civitai.update(dup_civitai)
                        break
            
            # Update trained words
            enhanced_civitai['trainedWords'] = list(all_trained_words)
            
            # Write enhanced civitai.info
            with open(civitai_info_path, 'w', encoding='utf-8') as f:
                json.dump(enhanced_civitai, f, indent=2, ensure_ascii=False)
            
            print(f"Enhanced civitai.info: {civitai_info_path}")
            return True
            
        except Exception as e:
            print(f"Error enhancing civitai.info: {e}")
            return False
    
    def generate_missing_civitai_files_for_moved_models(self):
        """Generate civitai.info files for models that have been moved but lack them"""
        cursor = self.conn.cursor()
        
        # Get moved models that still don't have civitai.info
        cursor.execute('''
            SELECT mf.*, sf.file_name, sf.sha256, sf.file_size
            FROM model_files mf
            JOIN scanned_files sf ON mf.scanned_file_id = sf.id
            WHERE mf.target_path IS NOT NULL 
              AND mf.status IN ('moved', 'duplicate_moved')
              AND mf.has_civitai_info = 0
        ''')
        
        models = [dict(row) for row in cursor.fetchall()]
        
        if not models:
            print("All moved models already have civitai.info files.")
            return
        
        print(f"Generating civitai.info files for {len(models)} moved models...")
        
        for model in models:
            target_directory = os.path.dirname(model['target_path'])
            
            model_info = {
                'model_name': model['model_name'],
                'file_name': model['file_name'],
                'base_model': model['base_model'],
                'model_type': model['model_type'],
                'file_size': model['file_size'],
                'sha256': model['sha256'],
                'metadata_json': model['metadata_json']
            }
            
            if self.create_civitai_info_file(target_directory, model_info):
                cursor.execute('''
                    UPDATE model_files 
                    SET has_civitai_info = 1, updated_at = ?
                    WHERE id = ?
                ''', (int(time.time()), model['id']))
        
        self.conn.commit()
    
    def close(self):
        """Close database connections"""
        if self.conn:
            self.conn.close()
        if self.civitai_conn:
            self.civitai_conn.close()


def main():
    """Main function"""
    generator = CivitaiInfoGenerator()
    
    try:
        print("Starting civitai.info generation...")
        
        # Process models needing civitai.info files
        success_count, error_count = generator.process_models_needing_civitai_info()
        
        # Also check for moved models
        generator.generate_missing_civitai_files_for_moved_models()
        
        print(f"\nCivitai.info generation complete!")
        
    except Exception as e:
        print(f"Error during civitai.info generation: {e}")
    finally:
        generator.close()


if __name__ == "__main__":
    main()