#!/usr/bin/env python3
"""
Metadata Extractor
Extracts metadata from model files and parses existing civitai.info/metadata.json files
"""

import json
import os
import sqlite3
import time
import configparser
from pathlib import Path
from typing import Dict, List, Optional, Any
import hashlib
import struct
import zipfile


class SafetensorsReader:
    """Reader for SafeTensors format metadata"""
    
    @staticmethod
    def read_metadata(file_path: str) -> Optional[Dict]:
        """Read metadata from safetensors file"""
        try:
            with open(file_path, 'rb') as f:
                # Read header length (first 8 bytes)
                header_size_bytes = f.read(8)
                if len(header_size_bytes) < 8:
                    return None
                
                header_size = struct.unpack('<Q', header_size_bytes)[0]
                
                # Read header JSON
                header_bytes = f.read(header_size)
                if len(header_bytes) < header_size:
                    return None
                
                header = json.loads(header_bytes.decode('utf-8'))
                
                # Extract metadata if it exists
                metadata = header.get('__metadata__', {})
                return metadata
        except Exception as e:
            safe_path = file_path.encode('utf-8', errors='replace').decode('utf-8')
            print(f"Error reading safetensors metadata from {safe_path}: {e}")
            return None


class MetadataExtractor:
    """Main metadata extraction class"""
    
    def safe_print_path(self, message: str, file_path: str):
        """Safely print file paths with Unicode characters"""
        if not self.verbose:
            return
            
        import sys
        
        try:
            # Get the terminal encoding, defaulting to utf-8
            encoding = sys.stdout.encoding or 'utf-8'
            
            # Truncate very long paths to prevent line wrapping issues
            max_path_length = 120
            if len(file_path) > max_path_length:
                truncated_path = file_path[:max_path_length-3] + "..."
            else:
                truncated_path = file_path
            
            # Try to encode the path for the terminal
            safe_path = truncated_path.encode(encoding, errors='replace').decode(encoding)
            print(f"{message}: {safe_path}")
            sys.stdout.flush()  # Ensure output is written immediately
        except (UnicodeError, AttributeError):
            # Fallback: replace problematic characters with placeholders
            safe_path = file_path.encode('ascii', errors='replace').decode('ascii')
            print(f"{message}: {safe_path}")
            sys.stdout.flush()
        except Exception:
            # Ultimate fallback
            print(f"{message}: <path with special characters>")
            sys.stdout.flush()
    
    def safe_path_repr(self, file_path: str) -> str:
        """Get a safe string representation of a file path"""
        import sys
        
        try:
            encoding = sys.stdout.encoding or 'utf-8'
            max_path_length = 120
            
            if len(file_path) > max_path_length:
                truncated_path = file_path[:max_path_length-3] + "..."
            else:
                truncated_path = file_path
            
            return truncated_path.encode(encoding, errors='replace').decode(encoding)
        except:
            return file_path.encode('ascii', errors='replace').decode('ascii')
    
    def __init__(self, db_path: str = "model_sorter.sqlite", civitai_db_path: str = "Database/civitai.sqlite", verbose: bool = True, config_path: str = "config.ini"):
        self.db_path = db_path
        self.civitai_db_path = civitai_db_path
        self.verbose = verbose
        self.conn = sqlite3.connect(db_path)
        self.safetensors_reader = SafetensorsReader()
        
        # Load configuration
        self.config = configparser.ConfigParser()
        if os.path.exists(config_path):
            self.config.read(config_path)
        
        # Get batch size from config or use default
        self.batch_size = self.config.getint('Logging', 'appended_model_metadata_interval', fallback=100)
        
        # Ensure model_files table exists
        self._create_tables()
        
        # Connect to civitai database if available
        self.civitai_conn = None
        self.civitai_db_valid = False
        if os.path.exists(civitai_db_path):
            try:
                # Open in read-only mode to prevent accidental creation
                self.civitai_conn = sqlite3.connect(f"file:{civitai_db_path}?mode=ro", uri=True)
                self.civitai_conn.row_factory = sqlite3.Row
                
                # Verify the database has the expected tables
                cursor = self.civitai_conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                
                if 'model_files' in tables and 'models' in tables and 'model_versions' in tables:
                    self.civitai_db_valid = True
                    print(f"Metadata extractor connected to civitai database: {civitai_db_path}")
                else:
                    print(f"Warning: Civitai database exists but missing required tables. Found tables: {tables}")
                    self.civitai_conn.close()
                    self.civitai_conn = None
                    
            except Exception as e:
                print(f"Warning: Could not connect to civitai database: {e}")
                if self.civitai_conn:
                    self.civitai_conn.close()
                    self.civitai_conn = None
    
    def _create_tables(self):
        """Create necessary tables if they don't exist"""
        cursor = self.conn.cursor()
        
        # Create model_files table matching file_scanner.py schema
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scanned_file_id INTEGER NOT NULL,
                model_name TEXT NOT NULL,
                base_model TEXT,
                model_type TEXT,
                civitai_id INTEGER,
                version_id INTEGER,
                source_path TEXT NOT NULL,
                target_path TEXT,
                is_duplicate BOOLEAN DEFAULT 0,
                duplicate_group_id INTEGER,
                metadata_json TEXT,
                has_civitai_info BOOLEAN DEFAULT 0,
                has_metadata_json BOOLEAN DEFAULT 0,
                status TEXT DEFAULT 'pending',
                error_message TEXT,
                created_at INTEGER DEFAULT (strftime('%s', 'now')),
                updated_at INTEGER DEFAULT (strftime('%s', 'now'))
            )
        ''')
        
        self.conn.commit()
    
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
                return None
            
            # Access by index since sqlite3 returns tuples by default
            file_id, model_id, version_id, file_type, sha256_hash, file_data = file_row
            
            # Get model info
            cursor.execute('''
                SELECT m.id, m.name, m.type, m.data
                FROM models m
                WHERE m.id = ?
            ''', (model_id,))
            
            model_row = cursor.fetchone()
            if not model_row:
                return None
            
            model_id_db, model_name, model_type, model_data = model_row
            
            # Get version info
            cursor.execute('''
                SELECT mv.id, mv.name, mv.base_model, mv.data
                FROM model_versions mv
                WHERE mv.id = ?
            ''', (version_id,))
            
            version_row = cursor.fetchone()
            if not version_row:
                return None
            
            version_id_db, version_name, base_model, version_data_str = version_row
            version_data = json.loads(version_data_str) if version_data_str else {}
            
            return {
                'name': model_name,
                'type': model_type,
                'base_model': base_model,
                'version_name': version_name,
                'trained_words': version_data.get('trainedWords', []),
                'civitai_id': model_id,
                'version_id': version_id
            }
            
        except Exception as e:
            print(f"Error looking up model in civitai database: {e}")
            return None
    
    def parse_civitai_info(self, file_path: str) -> Optional[Dict]:
        """Parse a .civitai.info file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            safe_path = file_path.encode('utf-8', errors='replace').decode('utf-8')
            print(f"Error parsing civitai.info file {safe_path}: {e}")
            return None
    
    def parse_metadata_json(self, file_path: str) -> Optional[Dict]:
        """Parse a .metadata.json file with fallback for malformed JSON"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except json.JSONDecodeError as e:
            # Try to extract just the JSON part if there's extra data
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Find the first complete JSON object
                decoder = json.JSONDecoder()
                idx = 0
                while idx < len(content):
                    try:
                        obj, end_idx = decoder.raw_decode(content, idx)
                        return obj  # Return the first valid JSON object
                    except json.JSONDecodeError:
                        idx += 1
                
                # If that fails, try line by line to find valid JSON
                lines = content.split('\n')
                for i in range(len(lines)):
                    try:
                        partial_content = '\n'.join(lines[:i+1])
                        return json.loads(partial_content)
                    except json.JSONDecodeError:
                        continue
                
                # Last resort: log the error but continue processing
                safe_path = file_path.encode('utf-8', errors='replace').decode('utf-8')
                print(f"Warning: Could not parse metadata.json (malformed JSON): {safe_path}")
                return None
                
            except Exception:
                safe_path = file_path.encode('utf-8', errors='replace').decode('utf-8')
                print(f"Warning: Could not parse metadata.json: {safe_path}")
                return None
        except Exception as e:
            safe_path = file_path.encode('utf-8', errors='replace').decode('utf-8')
            print(f"Error reading metadata.json file {safe_path}: {e}")
            return None
    
    def parse_readme_file(self, file_path: str) -> Optional[Dict]:
        """Parse README.md file and extract civitai information"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            import re
            extracted_data = {}
            
            # Extract Author
            author_match = re.search(r'Author:\s*\[([^\]]+)\]\(([^)]+)\)', content)
            if author_match:
                extracted_data['author'] = {
                    'name': author_match.group(1),
                    'url': author_match.group(2)
                }
            
            # Extract Model URL and IDs
            model_match = re.search(r'Model:\s*\[([^\]]+)\]\(([^)]+)\)', content)
            if model_match:
                model_url = model_match.group(2)
                extracted_data['model_url'] = model_url
                
                # Extract model ID and version ID from URL
                model_id_match = re.search(r'/models/(\d+)', model_url)
                if model_id_match:
                    extracted_data['civitai_id'] = int(model_id_match.group(1))
                
                version_id_match = re.search(r'modelVersionId=(\d+)', model_url)
                if version_id_match:
                    extracted_data['version_id'] = int(version_id_match.group(1))
            
            # Extract Mirror URL
            mirror_match = re.search(r'Mirror:\s*\[([^\]]+)\]\(([^)]+)\)', content)
            if mirror_match:
                extracted_data['mirror_url'] = mirror_match.group(2)
            
            return extracted_data if extracted_data else None
            
        except Exception as e:
            safe_path = file_path.encode('utf-8', errors='replace').decode('utf-8')
            print(f"Error parsing README file {safe_path}: {e}")
            return None
    
    def extract_model_metadata(self, model_path: str) -> Dict:
        """Extract metadata from a model file"""
        # Start with filename as fallback model name
        default_model_name = os.path.splitext(os.path.basename(model_path))[0]
        
        metadata = {
            'file_path': model_path,
            'file_name': os.path.basename(model_path),
            'file_size': os.path.getsize(model_path),
            'format': None,
            'embedded_metadata': {},
            'base_model': None,
            'model_type': None,
            'trained_words': [],
            'model_name': default_model_name
        }
        
        # Determine format and extract metadata
        file_ext = os.path.splitext(model_path)[1].lower()
        
        if file_ext == '.safetensors':
            metadata['format'] = 'SafeTensor'
            embedded_meta = self.safetensors_reader.read_metadata(model_path)
            if embedded_meta:
                metadata['embedded_metadata'] = embedded_meta
                
                # Extract common fields
                metadata['base_model'] = embedded_meta.get('ss_base_model_version', embedded_meta.get('base_model'))
                
                # Extract model name with filename fallback
                extracted_name = embedded_meta.get('ss_output_name', embedded_meta.get('modelspec.title'))
                if extracted_name:
                    metadata['model_name'] = extracted_name
                
                # Try to extract trained words
                if 'ss_tag_frequency' in embedded_meta:
                    # LoRA training tags
                    tag_freq = json.loads(embedded_meta['ss_tag_frequency']) if isinstance(embedded_meta['ss_tag_frequency'], str) else embedded_meta['ss_tag_frequency']
                    if isinstance(tag_freq, dict):
                        for dataset_name, tags in tag_freq.items():
                            if isinstance(tags, dict):
                                metadata['trained_words'].extend(tags.keys())
                
                # Determine model type from metadata
                if 'ss_network_module' in embedded_meta:
                    metadata['model_type'] = 'LORA'
                elif 'ss_base_model_version' in embedded_meta:
                    metadata['model_type'] = 'LORA'
                else:
                    metadata['model_type'] = 'Checkpoint'
        
        elif file_ext in ['.ckpt', '.pt', '.pth']:
            metadata['format'] = 'PyTorch'
            # For PyTorch files, we can't easily extract metadata without loading
            # the full model, so we'll rely on associated files
            
        elif file_ext == '.bin':
            metadata['format'] = 'Hugging Face'
            # Similar limitation for .bin files
        
        # If no model name found, use filename
        if not metadata['model_name']:
            metadata['model_name'] = os.path.splitext(os.path.basename(model_path))[0]
        
        return metadata
    
    def find_associated_metadata(self, model_path: str) -> Dict:
        """Find and parse associated metadata files for a model"""
        model_dir = os.path.dirname(model_path)
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        
        metadata_files = {
            'civitai_info': None,
            'metadata_json': None,
            'readme_info': None,
            'other_text_files': []
        }
        
        # Look for .civitai.info file
        civitai_info_path = os.path.join(model_dir, f"{model_name}.civitai.info")
        if os.path.exists(civitai_info_path):
            metadata_files['civitai_info'] = {
                'path': civitai_info_path,
                'data': self.parse_civitai_info(civitai_info_path)
            }
        
        # Look for .metadata.json file
        metadata_json_path = os.path.join(model_dir, f"{model_name}.metadata.json")
        if os.path.exists(metadata_json_path):
            metadata_files['metadata_json'] = {
                'path': metadata_json_path,
                'data': self.parse_metadata_json(metadata_json_path)
            }
        
        # Look for README.md file (especially useful if only one model in directory)
        readme_path = os.path.join(model_dir, "README.md")
        if os.path.exists(readme_path):
            # Check if there's only one model file in directory (or close to it)
            model_files = [f for f in os.listdir(model_dir) if f.endswith(('.safetensors', '.ckpt', '.pt', '.pth'))]
            if len(model_files) <= 2:  # Allow for some flexibility
                metadata_files['readme_info'] = {
                    'path': readme_path,
                    'data': self.parse_readme_file(readme_path)
                }
        
        # Look for other text files that might contain metadata
        for file_name in os.listdir(model_dir):
            if file_name.startswith(model_name) and file_name.endswith(('.txt', '.yaml', '.yml', '.json')):
                file_path = os.path.join(model_dir, file_name)
                if os.path.isfile(file_path) and file_path not in [civitai_info_path, metadata_json_path]:
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read(1000)  # Read first 1000 chars
                        metadata_files['other_text_files'].append({
                            'path': file_path,
                            'content_preview': content
                        })
                    except Exception:
                        pass
        
        return metadata_files
    
    def extract_base_model_from_metadata(self, model_metadata: Dict, associated_metadata: Dict, skip_civitai_lookup: bool = False) -> Optional[str]:
        """Extract base model information from various metadata sources"""
        base_model = None
        
        # Priority order: civitai.info > metadata.json > embedded metadata
        if associated_metadata.get('civitai_info') and associated_metadata['civitai_info'].get('data'):
            civitai_data = associated_metadata['civitai_info']['data']
            base_model = civitai_data.get('baseModel')
        
        if not base_model and associated_metadata.get('metadata_json') and associated_metadata['metadata_json'].get('data'):
            metadata_json = associated_metadata['metadata_json']['data']
            if metadata_json and isinstance(metadata_json, dict):
                base_model = metadata_json.get('base_model')
                if not base_model and 'civitai' in metadata_json:
                    civitai_section = metadata_json['civitai']
                    if isinstance(civitai_section, dict):
                        base_model = civitai_section.get('baseModel')
        
        # Check README.md for civitai information if still no base model
        if not base_model and associated_metadata['readme_info'] and associated_metadata['readme_info']['data']:
            readme_data = associated_metadata['readme_info']['data']
            if 'civitai_id' in readme_data:
                # Try to look up base model from civitai database using the model ID
                if self.civitai_conn:
                    try:
                        cursor = self.civitai_conn.cursor()
                        cursor.execute('''
                            SELECT mv.base_model 
                            FROM models m 
                            JOIN model_versions mv ON m.id = mv.model_id 
                            WHERE m.id = ? AND mv.id = ?
                        ''', (readme_data['civitai_id'], readme_data.get('version_id', readme_data['civitai_id'])))
                        
                        result = cursor.fetchone()
                        if result:
                            base_model = result[0]
                    except Exception as e:
                        print(f"Error looking up base model from README civitai ID: {e}")
        
        if not base_model:
            base_model = model_metadata.get('base_model')
        
        # If base model is still unknown, "Other", or "Unknown", try to extract from folder path
        # This should override Unknown/Other values regardless of civitai database status
        if not base_model or base_model in ['Other', 'Unknown', None]:
            file_path = model_metadata.get('file_path', '')
            path_base_model = self.extract_base_model_from_path(file_path)
            if path_base_model:
                base_model = path_base_model
        
        # Normalize base model names
        if base_model:
            base_model = self.normalize_base_model_name(base_model)
        
        return base_model
    
    def extract_base_model_from_path(self, file_path: str) -> Optional[str]:
        """Extract base model from folder path indicators"""
        if not file_path:
            return None
        
        path_upper = file_path.upper()
        
        # Check for base model indicators in path
        # Order by specificity (more specific first)
        base_model_indicators = [
            ('Flux.1 D', ['FLUX', 'FLUX.1', 'FLUX1']),
            ('SDXL 1.0', ['SDXL', 'SD XL', 'STABLE DIFFUSION XL']),
            ('SD 3.5', ['SD 3.5', 'SD3.5', 'STABLE DIFFUSION 3.5']),
            ('SD 3', ['SD 3', 'SD3', 'STABLE DIFFUSION 3']),
            ('SD 2.1', ['SD 2.1', 'SD2.1', 'STABLE DIFFUSION 2.1']),
            ('SD 2.0', ['SD 2.0', 'SD2.0', 'STABLE DIFFUSION 2.0']),
            ('SD 1.5', ['SD 1.5', 'SD1.5', 'STABLE DIFFUSION 1.5']),
            ('SD 1.4', ['SD 1.4', 'SD1.4', 'STABLE DIFFUSION 1.4']),
            ('Pony', ['PONY', 'PONYXL']),
            ('Illustrious', ['ILLUSTRIOUS']),
        ]
        
        for base_model, indicators in base_model_indicators:
            for indicator in indicators:
                if indicator in path_upper:
                    return base_model
        
        return None
    
    def normalize_base_model_name(self, base_model: Optional[str]) -> str:
        """Normalize base model names to match civitai database values"""
        if not base_model or base_model is None:
            return "Other"
        
        base_model_input = str(base_model).strip()
        base_model_lower = base_model_input.lower()
        
        # Get valid base models from civitai database (cache them)
        if not hasattr(self, '_valid_base_models'):
            self._valid_base_models = self._get_valid_base_models()
        
        # First, try exact match (case insensitive)
        for valid_model in self._valid_base_models:
            if base_model_lower == valid_model.lower():
                return valid_model
        
        # Then try fuzzy matching for common variations
        for valid_model in self._valid_base_models:
            valid_lower = valid_model.lower()
            
            # Handle common variations
            if any(keyword in base_model_lower for keyword in self._get_model_keywords(valid_lower)):
                return valid_model
        
        # If no match found, return original but title case
        return base_model_input.title()
    
    def _get_valid_base_models(self) -> List[str]:
        """Get list of valid base models from civitai database"""
        try:
            if self.civitai_conn:
                cursor = self.civitai_conn.cursor()
                cursor.execute("SELECT DISTINCT base_model FROM model_versions ORDER BY base_model")
                return [row[0] for row in cursor.fetchall()]
        except Exception:
            pass
        
        # Fallback to common base models if database query fails
        return ["SD 1.5", "SD 2.1", "SD 2.0", "SD 3", "SD 3.5", "SDXL 1.0", "Flux.1 D", "Pony", "Illustrious", "Other"]
    
    def _get_model_keywords(self, valid_model_lower: str) -> List[str]:
        """Get keywords that should match to a valid model"""
        keywords = [valid_model_lower]
        
        # Add common variations
        if 'sd 1.5' in valid_model_lower:
            keywords.extend(['sd1.5', 'stable diffusion 1.5', 'sd15'])
        elif 'sd 2.1' in valid_model_lower:
            keywords.extend(['sd2.1', 'stable diffusion 2.1', 'sd21'])
        elif 'sd 2.0' in valid_model_lower:
            keywords.extend(['sd2.0', 'stable diffusion 2.0', 'sd20'])
        elif 'sd 3.5' in valid_model_lower:
            keywords.extend(['sd3.5', 'stable diffusion 3.5', 'sd35'])
        elif 'sd 3' in valid_model_lower:
            keywords.extend(['sd3', 'stable diffusion 3'])
        elif 'sdxl' in valid_model_lower:
            keywords.extend(['sd xl', 'stable diffusion xl', 'sdxl'])
        elif 'flux' in valid_model_lower:
            keywords.extend(['flux', 'flux.1', 'flux1'])
        elif 'pony' in valid_model_lower:
            keywords.extend(['pony', 'ponyxl'])
        elif 'illustrious' in valid_model_lower:
            keywords.extend(['illustrious'])
            
        return keywords
    
    def extract_model_type_from_metadata(self, model_metadata: Dict, associated_metadata: Dict) -> str:
        """Extract model type from various metadata sources"""
        model_type = "LORA"  # Default assumption
        
        # Check civitai.info first
        if associated_metadata['civitai_info'] and associated_metadata['civitai_info']['data']:
            civitai_data = associated_metadata['civitai_info']['data']
            if 'model' in civitai_data and 'type' in civitai_data['model']:
                model_type = civitai_data['model']['type']
        
        # Check metadata.json
        if not model_type or model_type == "LORA":
            if associated_metadata['metadata_json'] and associated_metadata['metadata_json']['data']:
                metadata_json = associated_metadata['metadata_json']['data']
                if metadata_json and isinstance(metadata_json, dict) and 'civitai' in metadata_json:
                    civitai_section = metadata_json['civitai']
                    if isinstance(civitai_section, dict) and 'model' in civitai_section:
                        model_type = civitai_section['model'].get('type', model_type)
        
        # Check README.md for civitai information
        if not model_type or model_type == "LORA":
            if associated_metadata['readme_info'] and associated_metadata['readme_info']['data']:
                readme_data = associated_metadata['readme_info']['data']
                if 'civitai_id' in readme_data:
                    # Try to look up model type from civitai database using the model ID
                    if self.civitai_conn:
                        try:
                            cursor = self.civitai_conn.cursor()
                            cursor.execute('''
                                SELECT m.type 
                                FROM models m 
                                WHERE m.id = ?
                            ''', (readme_data['civitai_id'],))
                            
                            result = cursor.fetchone()
                            if result:
                                model_type = result[0]
                        except Exception as e:
                            print(f"Error looking up model type from README civitai ID: {e}")
        
        # Check embedded metadata
        if not model_type or model_type == "LORA":
            model_type = model_metadata.get('model_type', 'LORA')
        
        # Check civitai database directly by SHA256 if we still don't have a definitive type
        if not model_type or model_type == "LORA":
            sha256 = model_metadata.get('sha256')
            if sha256:
                civitai_match = self.lookup_model_in_civitai_db(sha256)
                if civitai_match and 'type' in civitai_match:
                    model_type = civitai_match['type']
        
        return model_type.upper() if model_type else "LORA"
    
    def create_comprehensive_metadata(self, model_path: str, skip_civitai_lookup: bool = False) -> Dict:
        """Create comprehensive metadata combining all sources"""
        # Extract model metadata
        model_metadata = self.extract_model_metadata(model_path)
        
        # Find associated metadata files
        associated_metadata = self.find_associated_metadata(model_path)
        
        # Check civitai database if we have SHA256 and not skipping
        civitai_db_info = None
        if not skip_civitai_lookup and hasattr(self, 'current_sha256') and self.current_sha256:
            civitai_db_info = self.lookup_model_in_civitai_db(self.current_sha256)
            if civitai_db_info:
                print(f"  Found in civitai database - correcting metadata")
        
        # Extract key information (civitai DB takes priority)
        if civitai_db_info:
            base_model = self.normalize_base_model_name(civitai_db_info['base_model'])
            model_type = civitai_db_info['type']
            model_name = civitai_db_info['name']
            trained_words = civitai_db_info.get('trained_words', [])
            civitai_id = civitai_db_info.get('civitai_id')
            version_id = civitai_db_info.get('version_id')
        else:
            base_model = self.extract_base_model_from_metadata(model_metadata, associated_metadata, skip_civitai_lookup)
            model_type = self.extract_model_type_from_metadata(model_metadata, associated_metadata)
        
            # Get trained words/trigger words (civitai DB already handled above)
            trained_words = model_metadata.get('trained_words', [])
            if associated_metadata['civitai_info'] and associated_metadata['civitai_info']['data']:
                civitai_data = associated_metadata['civitai_info']['data']
                if 'trainedWords' in civitai_data:
                    trained_words.extend(civitai_data['trainedWords'])
            
            # Remove duplicates from trained words
            trained_words = list(set(trained_words))
            
            # Get model name (civitai DB already handled above)
            model_name = model_metadata['model_name']
            if associated_metadata['civitai_info'] and associated_metadata['civitai_info']['data']:
                civitai_data = associated_metadata['civitai_info']['data']
                if 'model' in civitai_data and 'name' in civitai_data['model']:
                    model_name = civitai_data['model']['name']
            
            # Get civitai IDs (civitai DB already handled above)
            civitai_id = None
            version_id = None
            if associated_metadata['civitai_info'] and associated_metadata['civitai_info']['data']:
                civitai_data = associated_metadata['civitai_info']['data']
                civitai_id = civitai_data.get('modelId')
                version_id = civitai_data.get('id')
            elif associated_metadata['metadata_json'] and associated_metadata['metadata_json']['data']:
                metadata_json = associated_metadata['metadata_json']['data']
                if 'civitai' in metadata_json and metadata_json['civitai'] is not None:
                    civitai_id = metadata_json['civitai'].get('modelId')
                    version_id = metadata_json['civitai'].get('id')
            elif associated_metadata['readme_info'] and associated_metadata['readme_info']['data']:
                readme_data = associated_metadata['readme_info']['data']
                civitai_id = readme_data.get('civitai_id')
                version_id = readme_data.get('version_id')
        
        return {
            'file_path': model_path,
            'file_name': os.path.basename(model_path),
            'model_name': model_name,
            'base_model': base_model or "Other",
            'model_type': model_type,
            'trained_words': trained_words,
            'civitai_id': civitai_id,
            'version_id': version_id,
            'file_format': model_metadata['format'],
            'file_size': model_metadata['file_size'],
            'has_civitai_info': associated_metadata['civitai_info'] is not None,
            'has_metadata_json': associated_metadata['metadata_json'] is not None,
            'has_readme_info': associated_metadata['readme_info'] is not None,
            'embedded_metadata': model_metadata['embedded_metadata'],
            'civitai_data': associated_metadata['civitai_info']['data'] if associated_metadata['civitai_info'] else None,
            'metadata_json_data': associated_metadata['metadata_json']['data'] if associated_metadata['metadata_json'] else None,
            'readme_data': associated_metadata['readme_info']['data'] if associated_metadata['readme_info'] else None,
            'associated_files': associated_metadata
        }
    
    def update_model_record(self, scanned_file_id: int, comprehensive_metadata: Dict):
        """Update the model_files table with extracted metadata"""
        cursor = self.conn.cursor()
        
        # Insert or update model record
        cursor.execute('''
            INSERT OR REPLACE INTO model_files 
            (scanned_file_id, model_name, base_model, model_type, civitai_id, version_id,
             source_path, metadata_json, has_civitai_info, has_metadata_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            scanned_file_id,
            comprehensive_metadata['model_name'],
            comprehensive_metadata['base_model'],
            comprehensive_metadata['model_type'],
            comprehensive_metadata['civitai_id'],
            comprehensive_metadata['version_id'],
            comprehensive_metadata['file_path'],
            json.dumps(comprehensive_metadata),
            comprehensive_metadata['has_civitai_info'],
            comprehensive_metadata['has_metadata_json'],
            int(os.path.getmtime(comprehensive_metadata['file_path']))
        ))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def process_scanned_models(self):
        """Process scanned model files by cross-referencing with civitai database"""
        cursor = self.conn.cursor()
        
        # Step 1: Get ALL scanned models with any existing metadata in ONE query
        print("Loading scanned models and existing metadata...")
        cursor.execute('''
            SELECT sf.id, sf.file_path, sf.sha256, sf.file_size,
                   mf.base_model, mf.model_name, mf.model_type
            FROM scanned_files sf
            LEFT JOIN model_files mf ON sf.id = mf.scanned_file_id
            WHERE sf.file_type = 'model'
        ''')
        
        all_models = cursor.fetchall()
        
        # Step 2: Separate models into categories
        already_processed = []
        need_civitai_check = []
        need_full_processing = []
        
        for scanned_id, file_path, sha256, file_size, base_model, model_name, model_type in all_models:
            if base_model is not None:  # Already processed locally
                already_processed.append((scanned_id, file_path))
            else:
                need_civitai_check.append((scanned_id, file_path, sha256, file_size))
        
        print(f"Found {len(already_processed)} already processed models")
        print(f"Need to check {len(need_civitai_check)} models against civitai database")
        
        # Step 3: Bulk check civitai database for remaining models (in chunks)
        civitai_found = set()
        if need_civitai_check and self.civitai_conn and self.civitai_db_valid:
            print("Bulk checking civitai database...")
            civitai_cursor = self.civitai_conn.cursor()
            
            # Process in chunks to avoid SQLite variable limit (999 max variables)
            sha256_list = [sha256.upper() for _, _, sha256, _ in need_civitai_check]
            chunk_size = 900  # Safe chunk size under SQLite limit
            
            for i in range(0, len(sha256_list), chunk_size):
                chunk = sha256_list[i:i + chunk_size]
                placeholders = ','.join(['?' for _ in chunk])
                
                civitai_cursor.execute(f'''
                    SELECT sha256 FROM model_files 
                    WHERE sha256 IN ({placeholders})
                ''', chunk)
                
                chunk_found = {row[0] for row in civitai_cursor.fetchall()}
                civitai_found.update(chunk_found)
                
                if i % (chunk_size * 10) == 0:  # Progress every 10 chunks
                    print(f"  Checked {min(i + chunk_size, len(sha256_list))}/{len(sha256_list)} models...")
            
            print(f"Found {len(civitai_found)} models in civitai database")
        
        # Step 4: Categorize remaining models
        for scanned_id, file_path, sha256, file_size in need_civitai_check:
            if sha256.upper() in civitai_found:
                already_processed.append((scanned_id, file_path))  # Found in civitai
            else:
                need_full_processing.append((scanned_id, file_path, sha256, file_size))
        
        print(f"Final processing needed: {len(need_full_processing)} models")
        print(f"Batch size for metadata commits: {self.batch_size} models")
        
        # Step 5: Process only the models that truly need processing
        processed_count = 0
        skipped_count = len(already_processed)
        batch_results = []
        
        total_files = len(all_models)
        for i, (scanned_file_id, file_path, sha256, file_size) in enumerate(need_full_processing, 1):
            try:
                if self.verbose:
                    self.safe_print_path("Processing (not in civitai database)", file_path)
                elif i % 10 == 0:  # Show progress every 10 processed files when not verbose
                    print(f"Progress: {i}/{len(need_full_processing)} processed ({skipped_count} skipped)")
                processed_count += 1
                
                # Set current SHA256 for civitai database lookup
                self.current_sha256 = sha256
                
                # Extract comprehensive metadata (skip civitai lookup since we know it's not there)
                metadata = self.create_comprehensive_metadata(file_path, skip_civitai_lookup=True)
                
                # Add SHA256 and file size from scanned files
                metadata['sha256'] = sha256
                metadata['file_size'] = file_size
                metadata['scanned_file_id'] = scanned_file_id
                
                # Debug: Show what was extracted from metadata.json (only in verbose mode)
                if self.verbose and metadata['has_metadata_json'] and metadata['metadata_json_data']:
                    print(f"  DEBUG - metadata.json contains: {list(metadata['metadata_json_data'].keys())}")
                    if 'base_model' in metadata['metadata_json_data']:
                        print(f"  DEBUG - base_model field: {metadata['metadata_json_data']['base_model']}")
                    if 'civitai' in metadata['metadata_json_data']:
                        print(f"  DEBUG - civitai section: {metadata['metadata_json_data']['civitai']}")
                
                # Add to batch results
                batch_results.append(metadata)
                
                # Display metadata
                if self.verbose:
                    print(f"  Model: {metadata['model_name']}")
                    print(f"  Base Model: {metadata['base_model']}")
                    print(f"  Type: {metadata['model_type']}")
                    print(f"  Has civitai.info: {metadata['has_civitai_info']}")
                    print(f"  Has metadata.json: {metadata['has_metadata_json']}")
                    if metadata['trained_words']:
                        print(f"  Trained words: {', '.join(metadata['trained_words'][:5])}{'...' if len(metadata['trained_words']) > 5 else ''}")
                
                # Batch commit based on config parameter
                if len(batch_results) >= self.batch_size:
                    self._commit_batch_results(batch_results)
                    print(f"📊 Batch commit: {len(batch_results)} models added to database")
                    batch_results.clear()
                
            except Exception as e:
                self.safe_print_path("Error processing", file_path)
                print(f"  Error details: {e}")
                continue
        
        # Commit any remaining results in the final batch
        if batch_results:
            self._commit_batch_results(batch_results)
            print(f"📊 Final batch commit: {len(batch_results)} models added to database")
        
        print(f"\n📊 Metadata extraction complete:")
        print(f"   Processed (not in civitai): {processed_count}")
        print(f"   Skipped (found in civitai): {skipped_count}")
        print(f"   Total files: {total_files}")
    
    def _commit_batch_results(self, batch_results: List[Dict]):
        """Commit a batch of metadata results to the database"""
        cursor = self.conn.cursor()
        
        for metadata in batch_results:
            try:
                # Prepare metadata JSON
                metadata_json = {
                    'trained_words': metadata['trained_words'],
                    'civitai_data': metadata.get('civitai_data'),
                    'metadata_json_data': metadata.get('metadata_json_data'),
                    'embedded_metadata': metadata.get('embedded_metadata')
                }
                
                # Insert or update model_files record using schema from file_scanner.py
                cursor.execute('''
                    INSERT OR REPLACE INTO model_files (
                        scanned_file_id, model_name, base_model, model_type, civitai_id,
                        version_id, source_path, metadata_json, has_civitai_info, 
                        has_metadata_json, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metadata['scanned_file_id'],
                    metadata['model_name'],
                    metadata['base_model'],
                    metadata['model_type'],
                    metadata.get('civitai_id'),
                    metadata.get('version_id'),
                    metadata['file_path'],  # source_path
                    json.dumps(metadata_json),  # metadata as JSON string
                    metadata['has_civitai_info'],
                    metadata['has_metadata_json'],
                    int(time.time())  # current timestamp
                ))
                
                # Get the model_file_id we just created/updated
                model_file_id = cursor.lastrowid
                if not model_file_id:
                    # For UPDATE operations, get the existing ID
                    cursor.execute('SELECT id FROM model_files WHERE scanned_file_id = ?', 
                                 (metadata['scanned_file_id'],))
                    result = cursor.fetchone()
                    if result:
                        model_file_id = result[0]
                
                # Transfer temporary associations to associated_files table
                if model_file_id:
                    self._transfer_temp_associations(cursor, metadata['scanned_file_id'], model_file_id)
            except Exception as e:
                print(f"Error inserting metadata for {metadata.get('file_path', 'unknown')}: {e}")
                continue
        
        self.conn.commit()
    
    def _transfer_temp_associations(self, cursor, model_scanned_file_id: int, model_file_id: int) -> None:
        """Transfer temporary associations from file scanning to the associated_files table"""
        try:
            # Check if temp_file_associations table exists
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='temp_file_associations'
            """)
            
            if not cursor.fetchone():
                return  # No temp table, nothing to transfer
            
            # Transfer associations from temp table to associated_files
            cursor.execute('''
                INSERT OR REPLACE INTO associated_files 
                (model_file_id, scanned_file_id, association_type, source_path, is_moved)
                SELECT ?, assoc_scanned_file_id, association_type, source_path, 0
                FROM temp_file_associations
                WHERE model_scanned_file_id = ?
            ''', (model_file_id, model_scanned_file_id))
            
            # Remove transferred associations from temp table
            cursor.execute('''
                DELETE FROM temp_file_associations 
                WHERE model_scanned_file_id = ?
            ''', (model_scanned_file_id,))
            
        except Exception as e:
            print(f"Warning: Could not transfer associations for model {model_scanned_file_id}: {e}")
    
    def close(self):
        """Close database connections"""
        if self.conn:
            self.conn.close()
        if hasattr(self, 'civitai_conn') and self.civitai_conn:
            self.civitai_conn.close()


def main():
    """Main function"""
    # Verify civitai database exists in expected location
    civitai_path = "Database/civitai.sqlite"
    if not os.path.exists(civitai_path):
        print(f"Warning: Civitai database not found at {civitai_path}")
        print("The script will process all models without civitai cross-reference")
    
    extractor = MetadataExtractor()
    
    try:
        extractor.process_scanned_models()
        print("Metadata extraction complete!")
        
    except Exception as e:
        print(f"Error during metadata extraction: {e}")
    finally:
        extractor.close()


if __name__ == "__main__":
    main()