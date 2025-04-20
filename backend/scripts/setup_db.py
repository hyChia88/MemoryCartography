# setup_db.py
import sqlite3
import os

def setup_database():
    db_file = 'memories.db'
    
    # Check if the database file exists
    db_exists = os.path.exists(db_file)
    
    # Connect to database (creates it if it doesn't exist)
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    # Create the memories table if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS memories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        location TEXT NOT NULL,
        date TEXT NOT NULL,
        type TEXT NOT NULL,
        keywords TEXT,
        content TEXT
    )
    ''')
    
    # Check existing columns
    cursor.execute("PRAGMA table_info(memories)")
    columns = [column[1] for column in cursor.fetchall()]
    
    # Add missing columns
    if 'description' not in columns:
        print("Adding 'description' column...")
        cursor.execute("ALTER TABLE memories ADD COLUMN description TEXT")
    
    if 'filename' not in columns:
        print("Adding 'filename' column...")
        cursor.execute("ALTER TABLE memories ADD COLUMN filename TEXT")
    
    if 'weight' not in columns:
        print("Adding 'weight' column...")
        cursor.execute("ALTER TABLE memories ADD COLUMN weight REAL DEFAULT 1.0")
    
    if 'image_path' not in columns:
        print("Adding 'image_path' column...")
        cursor.execute("ALTER TABLE memories ADD COLUMN image_path TEXT")
    
    if 'embedding' not in columns:
        print("Adding 'embedding' column...")
        cursor.execute("ALTER TABLE memories ADD COLUMN embedding BLOB")
    
    conn.commit()
    conn.close()
    
    print(f"Database setup completed. {'Created new database.' if not db_exists else 'Updated existing database.'}")

if __name__ == "__main__":
    setup_database()