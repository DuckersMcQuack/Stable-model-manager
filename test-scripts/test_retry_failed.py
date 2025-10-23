#!/usr/bin/env python3
"""
Test the retry-failed functionality with simulated column lacking errors
"""

import json
import os
from datetime import datetime

def create_test_column_lacking_file():
    """Create a test column_lacking.json file with simulated errors"""
    
    test_errors = [
        {
            "file_path": "sample_metadata/sample_image1.png",
            "column_name": "addnet_enabled", 
            "error_message": "table media_metadata has no column named addnet_enabled",
            "timestamp": datetime.now().isoformat(),
            "retry_count": 0
        },
        {
            "file_path": "sample_metadata/sample_image2.png",
            "column_name": "ensd",
            "error_message": "table media_metadata has no column named ensd", 
            "timestamp": datetime.now().isoformat(),
            "retry_count": 0
        }
    ]
    
    with open('column_lacking.json', 'w') as f:
        json.dump(test_errors, f, indent=2)
    
    print(f"Created test column_lacking.json with {len(test_errors)} simulated errors")
    return test_errors

if __name__ == "__main__":
    create_test_column_lacking_file()
    
    # Show the retry-failed command
    print("\nTo test retry functionality, run:")
    print("python model_sorter_main.py --retry-failed")
    print("\nOr with file_scanner.py:")  
    print("python file_scanner.py --retry-failed")