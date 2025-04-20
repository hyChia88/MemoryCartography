# backend/app/database.py
import sqlite3
import json
from datetime import datetime
from pathlib import Path

# Database initialization function
def init_db(db_path='memories.db'):
    """Initialize the SQLite database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # Create memories table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS memories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        location TEXT NOT NULL,
        date TEXT NOT NULL,
        type TEXT NOT NULL,
        keywords TEXT,
        description TEXT,
        filename TEXT,
        weight REAL DEFAULT 1.0,
        embedding TEXT
    )
    ''')

    conn.commit()
    conn.close()

def add_memory(filename, title, location, date, memory_type,
keywords=None, description=None, weight=1.0, embedding=None):
    """Add a new memory to the database."""
    conn = sqlite3.connect('memories.db')
    cursor = conn.cursor()
    # Serialize keywords and embedding to JSON
    keywords_json = json.dumps(keywords) if keywords else None
    embedding_json = json.dumps(embedding) if embedding else None

    cursor.execute(
        '''
        INSERT INTO memories 
        (filename, title, location, date, type, keywords, description, weight, embedding) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''',
        (filename, title, location, date, memory_type, keywords_json, 
        description, weight, embedding_json)
    )

    conn.commit()
    memory_id = cursor.lastrowid
    conn.close()

    return memory_id

def search_memories(query, memory_type='user', limit=10):
    """Search memories by query."""
    conn = sqlite3.connect('memories.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    # Simple search across title, location, and keywords
    cursor.execute(
        """
        SELECT * FROM memories 
        WHERE type = ? AND (
            title LIKE ? OR 
            location LIKE ? OR 
            keywords LIKE ?
        ) 
        ORDER BY weight DESC 
        LIMIT ?
        """, 
        (memory_type, f"%{query}%", f"%{query}%", f"%{query}%", limit)
    )

    memories = [dict(row) for row in cursor.fetchall()]

    # Deserialize keywords
    for memory in memories:
        memory['keywords'] = json.loads(memory['keywords']) if memory['keywords'] else []

    conn.close()
    return memories
