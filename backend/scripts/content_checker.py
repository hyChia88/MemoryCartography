#!/usr/bin/env python
"""
Database Content Checker for Memory Cartography

This script checks the content of your memories database and reports what's inside.
"""

import sqlite3
import os
import sys
import json

def inspect_database(db_path):
    """Check the content of the database and report details."""
    if not os.path.exists(db_path):
        print(f"Error: Database not found at {db_path}")
        return False
    
    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 1. Check table structure
        print("\n=== DATABASE SCHEMA ===")
        cursor.execute("PRAGMA table_info(memories)")
        columns = cursor.fetchall()
        
        if not columns:
            print("ERROR: No 'memories' table found in the database!")
            return False
            
        print(f"Found {len(columns)} columns in the 'memories' table:")
        for col in columns:
            col_id, name, type_name, not_null, default_val, pk = col
            print(f"  - {name} ({type_name}){' PRIMARY KEY' if pk else ''}{' NOT NULL' if not_null else ''}")
        
        # 2. Count records
        print("\n=== RECORD COUNTS ===")
        cursor.execute("SELECT COUNT(*) FROM memories")
        total_count = cursor.fetchone()[0]
        print(f"Total records in database: {total_count}")
        
        if total_count == 0:
            print("WARNING: Your database has no records!")
            return False
            
        # Check counts by type
        cursor.execute("SELECT type, COUNT(*) FROM memories GROUP BY type")
        type_counts = cursor.fetchall()
        
        for type_name, count in type_counts:
            print(f"Records of type '{type_name}': {count}")
        
        # 3. Sample records
        if total_count > 0:
            print("\n=== SAMPLE RECORDS ===")
            cursor.execute("SELECT * FROM memories LIMIT 3")
            sample_records = cursor.fetchall()
            
            for i, record in enumerate(sample_records):
                print(f"\nRecord #{i+1}:")
                for j, col in enumerate(columns):
                    col_name = col[1]
                    value = record[j]
                    
                    # Pretty-print JSON fields
                    if col_name in ['keywords', 'openai_keywords', 'detected_objects'] and value:
                        try:
                            parsed = json.loads(value)
                            value = json.dumps(parsed, indent=2)
                        except:
                            pass
                    
                    # Truncate long values
                    if isinstance(value, str) and len(value) > 100:
                        value = value[:100] + "..."
                        
                    print(f"  {col_name}: {value}")
        
        # 4. Check search-related columns
        print("\n=== SEARCH CAPABILITY CHECK ===")
        search_columns = ['description', 'openai_description', 'keywords', 'openai_keywords']
        for col_name in search_columns:
            has_col = any(col[1] == col_name for col in columns)
            if has_col:
                cursor.execute(f"SELECT COUNT(*) FROM memories WHERE {col_name} IS NOT NULL AND {col_name} != ''")
                non_empty = cursor.fetchone()[0]
                print(f"Column '{col_name}': Present - Contains data in {non_empty}/{total_count} records")
            else:
                print(f"Column '{col_name}': MISSING")
        
        # 5. Test search for 'happy'
        print("\n=== TEST SEARCH ===")
        search_term = "happy"
        
        # Try all available text columns
        text_columns = [col[1] for col in columns if col[2].upper() in ('TEXT', 'VARCHAR', 'STRING')]
        conditions = []
        
        for col in text_columns:
            conditions.append(f"LOWER({col}) LIKE '%{search_term.lower()}%'")
        
        if conditions:
            query = f"SELECT COUNT(*) FROM memories WHERE {' OR '.join(conditions)}"
            cursor.execute(query)
            match_count = cursor.fetchone()[0]
            
            print(f"Search for '{search_term}' in all text columns: Found {match_count} matches")
            
            if match_count > 0:
                print("\nMatching records:")
                cursor.execute(f"SELECT id, title, type FROM memories WHERE {' OR '.join(conditions)} LIMIT 5")
                for record in cursor.fetchall():
                    print(f"  ID: {record[0]}, Title: {record[1]}, Type: {record[2]}")
            
        return True
        
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return False
    finally:
        if 'conn' in locals():
            conn.close()

def main():
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    else:
        # Check the standard locations
        base_dir = os.path.abspath('.')
        
        # Potential database paths
        potential_paths = []
        
        # Check if we're in the app directory
        if os.path.basename(base_dir) == 'app':
            potential_paths.append(os.path.join(base_dir, 'data', 'metadata', 'memories.db'))
            potential_paths.append(os.path.join(base_dir, 'data', 'metadata_ext_featured', 'memories_with_visual_features.db'))
        elif os.path.exists(os.path.join(base_dir, 'app')):
            # We're in the parent directory
            potential_paths.append(os.path.join(base_dir, 'app', 'data', 'metadata', 'memories.db'))
            potential_paths.append(os.path.join(base_dir, 'app', 'data', 'metadata_ext_featured', 'memories_with_visual_features.db'))
        elif os.path.exists(os.path.join(base_dir, 'backend', 'app')):
            # We're in the project root
            potential_paths.append(os.path.join(base_dir, 'backend', 'app', 'data', 'metadata', 'memories.db'))
            potential_paths.append(os.path.join(base_dir, 'backend', 'app', 'data', 'metadata_ext_featured', 'memories_with_visual_features.db'))
        
        # Try all paths and use the first one that exists
        db_path = None
        for path in potential_paths:
            if os.path.exists(path):
                db_path = path
                break
        
        # If we didn't find a database, ask the user
        if not db_path:
            db_path = input("Please enter the full path to your memories.db file: ")
    
    print(f"Inspecting database at: {db_path}")
    inspect_database(db_path)

if __name__ == "__main__":
    main()