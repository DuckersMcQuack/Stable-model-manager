#!/usr/bin/env python3
"""
Database Inspector Tool
Analyzes the existing civitai.sqlite database structure and content
"""

import sqlite3
import json
import os
from pathlib import Path


def inspect_database(db_path):
    """Inspect the civitai.sqlite database and return structure info"""
    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}")
        return None
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        db_info = {
            "database_path": db_path,
            "tables": {}
        }
        
        print(f"Database: {db_path}")
        print(f"Found {len(tables)} tables:")
        
        for table_name in tables:
            table_name = table_name[0]
            print(f"\n--- Table: {table_name} ---")
            
            # Get table schema
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            
            # Get table row count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
            row_count = cursor.fetchone()[0]
            
            # Get sample rows (max 5)
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 5;")
            sample_rows = cursor.fetchall()
            
            table_info = {
                "columns": columns,
                "row_count": row_count,
                "sample_rows": sample_rows
            }
            
            db_info["tables"][table_name] = table_info
            
            print(f"Columns ({len(columns)}):")
            for col in columns:
                print(f"  {col[1]} ({col[2]}) - {col}")
            
            print(f"Row count: {row_count}")
            
            if sample_rows and row_count > 0:
                print("Sample rows:")
                column_names = [col[1] for col in columns]
                for i, row in enumerate(sample_rows):
                    print(f"  Row {i+1}:")
                    for j, value in enumerate(row):
                        if j < len(column_names):
                            # Truncate long values for readability
                            str_val = str(value)
                            if len(str_val) > 100:
                                str_val = str_val[:100] + "..."
                            print(f"    {column_names[j]}: {str_val}")
        
        conn.close()
        return db_info
        
    except Exception as e:
        print(f"Error inspecting database: {e}")
        return None


def main():
    # Get database path from config or use default
    config_path = Path("config.ini")
    db_path = "Database/civitai.sqlite"
    
    if config_path.exists():
        import configparser
        config = configparser.ConfigParser()
        config.read(config_path)
        db_path = config.get('Paths', 'database_path', fallback=db_path)
    
    # Inspect the database
    db_info = inspect_database(db_path)
    
    if db_info:
        # Save analysis to file
        analysis_file = "database_analysis.json"
        with open(analysis_file, 'w') as f:
            json.dump(db_info, f, indent=2, default=str)
        print(f"\nFull analysis saved to: {analysis_file}")


if __name__ == "__main__":
    main()