# backend/init_memories.py
import json
import sqlite3
import os
from pathlib import Path

def init_memories_from_json(json_path, memory_type='user'):
    """Initialize memories from a JSON file."""
    # Connect to SQLite database
    conn = sqlite3.connect('memories.db')
    cursor = conn.cursor()
    # Create memories table if not exists
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS memories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        title TEXT,
        location TEXT,
        date TEXT,
        type TEXT,
        keywords TEXT,
        description TEXT,
        weight REAL,
        embedding TEXT,
        original_path TEXT
    )
    ''')

    # Read JSON file
    with open(json_path, 'r') as f:
        memories_data = json.load(f)

    # Insert memories
    for filename, memory in memories_data.items():
        cursor.execute('''
        INSERT INTO memories 
        (filename, title, location, date, type, keywords, description, weight, embedding, original_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            filename,
            f"{memory['location']} - {filename}",  # Generate title
            memory['location'],
            memory['date'],
            memory_type,  # user or public
            json.dumps(memory.get('keywords', [])),
            memory.get('description', ''),
            memory.get('weight', 1.0),
            json.dumps(memory.get('embedding', [])) if memory.get('embedding') else None,
            memory.get('original_path', '')
        ))

    # Commit and close
    conn.commit()
    conn.close()
    print(f"Inserted {len(memories_data)} {memory_type} memories from {json_path}")
    
def get_sample_memories(limit=5):
    """Retrieve and print sample memories."""
    conn = sqlite3.connect('memories.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM memories LIMIT ?', (limit,))
    memories = cursor.fetchall()

    print("\n--- Sample Memories ---")
    for memory in memories:
        print(f"ID: {memory[0]}")
        print(f"Filename: {memory[1]}")
        print(f"Title: {memory[2]}")
        print(f"Location: {memory[3]}")
        print(f"Date: {memory[4]}")
        print(f"Type: {memory[5]}")
        print(f"Keywords: {memory[6]}")
        print(f"Original Path: {memory[9]}")
        print("---")

    conn.close()
    
def main():
    # Base path (adjust if needed)
    base_path = Path(__file__).parent.parent / 'data' / 'metadata'
    # Paths to JSON files
    user_metadata_path = base_path / 'user_metadata.json'
    public_metadata_path = base_path / 'public_metadata.json'

    # Initialize memories
    if user_metadata_path.exists():
        init_memories_from_json(user_metadata_path, 'user')
    else:
        print(f"User metadata file not found: {user_metadata_path}")

    if public_metadata_path.exists():
        init_memories_from_json(public_metadata_path, 'public')
    else:
        print(f"Public metadata file not found: {public_metadata_path}")

    # Get and print sample memories
    get_sample_memories()
    

print("init memories db")
main()