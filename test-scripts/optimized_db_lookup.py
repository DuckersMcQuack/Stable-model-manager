#!/usr/bin/env python3
"""
Optimized database lookup strategy - single query per file
"""

def optimized_file_lookup(self, file_path: str, file_name: str, file_size: int, last_modified: float):
    """
    Single query to check if file needs scanning
    Returns: (should_skip, existing_record)
    """
    cursor = self.conn.cursor()
    
    # Single query that checks everything at once
    cursor.execute('''
        SELECT id, file_path, file_name, file_size, sha256, autov3, 
               file_type, extension, last_modified, scan_date,
               image_metadata, has_image_metadata, created_at, updated_at
        FROM scanned_files 
        WHERE (file_path = ? AND last_modified = ?) 
           OR (file_name = ? AND file_size = ?)
        LIMIT 1
    ''', (file_path, last_modified, file_name, file_size))
    
    row = cursor.fetchone()
    if not row:
        return False, None  # File not in database - need to scan
    
    columns = [desc[0] for desc in cursor.description]
    existing_record = dict(zip(columns, row))
    
    # Check if it's an exact match (path + timestamp)
    if (existing_record['file_path'] == file_path and 
        existing_record['last_modified'] == last_modified):
        return True, existing_record  # Exact match - skip scanning
    
    # Check if it's just a name/size match (possible duplicate)
    if (existing_record['file_name'] == file_name and 
        existing_record['file_size'] == file_size):
        return True, existing_record  # Same file exists - skip scanning
    
    return False, None  # Need to scan


# Alternative: Even more optimized with prepared statement cache
class OptimizedFileChecker:
    def __init__(self, db_connection):
        self.conn = db_connection
        # Pre-prepare the query for better performance
        self.lookup_stmt = self.conn.cursor()
    
    def check_file_needs_scan(self, file_path: str, file_name: str, file_size: int, last_modified: float):
        """Ultra-fast single query file check"""
        
        # Use a single optimized query
        self.lookup_stmt.execute('''
            SELECT id, sha256, autov3, file_type, last_modified,
                   CASE 
                     WHEN file_path = ? AND last_modified = ? THEN 'exact_match'
                     WHEN file_name = ? AND file_size = ? THEN 'name_size_match' 
                     ELSE 'no_match'
                   END as match_type
            FROM scanned_files 
            WHERE file_path = ? OR (file_name = ? AND file_size = ?)
            ORDER BY match_type  -- Prioritize exact matches
            LIMIT 1
        ''', (file_path, last_modified, file_name, file_size, file_path, file_name, file_size))
        
        row = self.lookup_stmt.fetchone()
        return row