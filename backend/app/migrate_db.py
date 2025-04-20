# migrate_db.py
import sqlite3

def migrate_database():
    conn = sqlite3.connect('memories.db')
    cursor = conn.cursor()
    
    # Check existing columns
    cursor.execute("PRAGMA table_info(memories)")
    columns = [column[1] for column in cursor.fetchall()]
    
    # Add missing columns
    if 'description' not in columns:
        cursor.execute("ALTER TABLE memories ADD COLUMN description TEXT")
    if 'filename' not in columns:
        cursor.execute("ALTER TABLE memories ADD COLUMN filename TEXT")
    if 'weight' not in columns:
        cursor.execute("ALTER TABLE memories ADD COLUMN weight REAL DEFAULT 1.0")
    if 'image_path' not in columns:
        cursor.execute("ALTER TABLE memories ADD COLUMN image_path TEXT")
    
    conn.commit()
    conn.close()
    print("Database migration completed.")

if __name__ == "__main__":
    migrate_database()