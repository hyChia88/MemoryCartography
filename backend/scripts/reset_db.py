#!/usr/bin/env python
"""
Quick Database Column Fix for Memory Cartography

This script fixes the immediate 'no such column: description' error
by adding the missing column to your database.
"""

import sqlite3
import os
import sys

def fix_database_now(db_path):
    """Add the missing 'description' column to the database."""
    if not os.path.exists(db_path):
        print(f"Error: Database not found at {db_path}")
        return False
    
    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if the 'description' column exists
        cursor.execute("PRAGMA table_info(memories)")
        columns = [info[1] for info in cursor.fetchall()]
        
        # Check for the specific columns that are causing errors
        needed_columns = ['description', 'keywords', 'weight', 'embedding']
        missing_columns = [col for col in needed_columns if col not in columns]
        
        if not missing_columns:
            print("All required columns already exist in the database.")
            return True
        
        # Add missing columns
        for column in missing_columns:
            try:
                if column == 'weight':
                    cursor.execute(f"ALTER TABLE memories ADD COLUMN {column} REAL DEFAULT 1.0")
                else:
                    cursor.execute(f"ALTER TABLE memories ADD COLUMN {column} TEXT")
                print(f"Added missing column: {column}")
            except sqlite3.Error as e:
                print(f"Error adding column {column}: {e}")
        
        # Now, copy data from new columns to old columns if they exist
        column_mappings = {
            'openai_description': 'description',
            'openai_keywords': 'keywords',
            'impact_weight': 'weight',
            'resnet_embedding': 'embedding'
        }
        
        for src_col, dest_col in column_mappings.items():
            if src_col in columns and dest_col in columns + missing_columns:
                try:
                    cursor.execute(f"UPDATE memories SET {dest_col} = {src_col} WHERE {src_col} IS NOT NULL")
                    rows_updated = cursor.rowcount
                    print(f"Copied data from {src_col} to {dest_col} ({rows_updated} rows updated)")
                except sqlite3.Error as e:
                    print(f"Error copying data from {src_col} to {dest_col}: {e}")
        
        # Commit changes
        conn.commit()
        print(f"Database {db_path} updated successfully!")
        return True
        
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return False
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    else:
        # Try to intelligently find the database path
        base_dir = os.path.abspath('.')
        
        # Check if we're in the app directory
        if os.path.basename(base_dir) == 'app':
            db_path = os.path.join(base_dir, 'data', 'metadata', 'memories.db')
        elif os.path.exists(os.path.join(base_dir, 'app')):
            # We're in the parent directory
            db_path = os.path.join(base_dir, 'app', 'data', 'metadata', 'memories.db')
        elif os.path.exists(os.path.join(base_dir, 'backend', 'app')):
            # We're in the project root
            db_path = os.path.join(base_dir, 'backend', 'app', 'data', 'metadata', 'memories.db')
        else:
            # Ask the user for the path
            db_path = input("Please enter the full path to your memories.db file: ")
    
    # Handle the special case where memory-map is in the path
    if 'memory-map' in db_path and 'backend' not in db_path:
        db_path = db_path.replace('memory-map', 'memory-map/backend')
        print(f"Adjusted path to include backend directory: {db_path}")
    
    if not os.path.exists(db_path):
        print(f"Warning: Database not found at {db_path}")
        alt_paths = [
            db_path.replace('/metadata/', '/metadata_ext_featured/'),
            db_path.replace('memories.db', 'memories_with_visual_features.db'),
            os.path.join(os.path.dirname(db_path), 'memories_with_visual_features.db')
        ]
        
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                print(f"Found alternative database at: {alt_path}")
                use_alt = input(f"Use this database instead? (y/n): ")
                if use_alt.lower() == 'y':
                    db_path = alt_path
                    break
    
    if not os.path.exists(db_path):
        print(f"Error: Could not find a valid database. Please specify the path as an argument.")
        sys.exit(1)
    
    print(f"Fixing database at: {db_path}")
    if fix_database_now(db_path):
        print("Fix completed successfully!")
        print("You can now restart your application and it should work correctly.")
    else:
        print("Failed to fix the database. Please check the error messages above.")