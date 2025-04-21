import sqlite3
import os

def reset_database():
    # Define database path
    db_path = 'backend\data\metadata\memories.db'
    
    # Remove existing database if it exists
    if os.path.exists(db_path):
        print(f"Removing existing database at {db_path}")
        os.remove(db_path)
    
    # Create a fresh database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create memories table with ALL required columns
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS memories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        location TEXT NOT NULL,
        date TEXT NOT NULL,
        type TEXT NOT NULL,
        keywords TEXT,
        content TEXT,
        description TEXT,
        filename TEXT,
        weight REAL DEFAULT 1.0,
        image_path TEXT,
        embedding BLOB
    )
    ''')
    
    conn.commit()
    conn.close()
    
    print(f"Created fresh database at {db_path}")
    print("Now ready for seeding data.")

if __name__ == "__main__":
    reset_database()