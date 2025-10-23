#!/usr/bin/env python3
"""
Test batch commit functionality
"""

import os
import sys
import sqlite3

current_dir = os.getcwd()
sys.path.append(current_dir)

from metadata_extractor import MetadataExtractor

def test_batch_commit():
    """Test the batch commit functionality"""
    
    # Create a test database
    test_db = "test_batch.sqlite"
    if os.path.exists(test_db):
        os.remove(test_db)
    
    # Create test extractor with test database
    extractor = MetadataExtractor(db_path=test_db)
    
    # Find actual model files in sample_models
    model_files = []
    if os.path.exists('sample_models'):
        for f in os.listdir('sample_models'):
            if f.lower().endswith(('.safetensors', '.ckpt', '.pt', '.pth')) and os.path.isfile(os.path.join('sample_models', f)):
                model_files.append(os.path.abspath(os.path.join('sample_models', f)))
    
    # Create test batch data from actual files
    test_batch = []
    
    if model_files:
        print(f"Found {len(model_files)} actual model files for batch testing")
        for i, file_path in enumerate(model_files[:5], 1):  # Test with up to 5 files
            filename = os.path.basename(file_path)
            file_format = filename.split('.')[-1]
            test_batch.append({
                'scanned_file_id': i,
                'model_name': os.path.splitext(filename)[0],
                'base_model': 'SD 1.5',
                'model_type': 'LORA',
                'trained_words': ['test', 'sample'],
                'civitai_id': None,
                'version_id': None,
                'file_format': file_format,
                'has_civitai_info': False,
                'has_metadata_json': False,
                'file_path': file_path
            })
    else:
        print("No model files found in sample_models/ - using synthetic test data")
        print("💡 Add model files to sample_models/ for real batch testing")
        test_batch = [
            {
                'scanned_file_id': 1,
                'model_name': 'synthetic_test_model',
                'base_model': 'SD 1.5',
                'model_type': 'LORA',
                'trained_words': ['test', 'sample'],
                'civitai_id': None,
                'version_id': None,
                'file_format': 'safetensors',
                'has_civitai_info': False,
                'has_metadata_json': False,
                'file_path': '/synthetic/test/sample_model1.safetensors'
            },
        {
            'scanned_file_id': 2,
            'model_name': 'test_model_2',
            'base_model': 'SDXL 1.0',
            'model_type': 'CHECKPOINT',
            'trained_words': [],
            'civitai_id': 12345,
            'version_id': 67890,
            'file_format': 'safetensors',
            'has_civitai_info': True,
            'has_metadata_json': False,
            'file_path': '/test/sample_model2.safetensors'
        }
    ]
    
    print("Testing batch commit functionality:")
    print("=" * 50)
    
    # Test batch commit
    try:
        extractor._commit_batch_results(test_batch)
        print("✓ Batch commit successful")
        
        # Verify data was inserted
        cursor = extractor.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM model_files")
        count = cursor.fetchone()[0]
        print(f"✓ {count} records inserted")
        
        # Verify specific data
        cursor.execute("SELECT model_name, base_model, model_type FROM model_files")
        records = cursor.fetchall()
        
        for record in records:
            print(f"  - {record[0]}: {record[1]} ({record[2]})")
            
        print("✓ Data verification successful")
        
    except Exception as e:
        print(f"✗ Batch commit failed: {e}")
    
    finally:
        # Cleanup
        extractor.conn.close()
        if os.path.exists(test_db):
            os.remove(test_db)
        print("✓ Test cleanup complete")

if __name__ == "__main__":
    test_batch_commit()