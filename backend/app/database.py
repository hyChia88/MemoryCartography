# database.py
import sqlite3
import json
import os
from pathlib import Path

def init_db(db_path):
    """Initialize the SQLite database."""
    # Ensure the directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # Create memories table with all needed columns
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
        embedding TEXT,
        openai_keywords TEXT,
        openai_description TEXT,
        impact_weight REAL DEFAULT 1.0,
        resnet_embedding TEXT,
        detected_objects TEXT
    )
    ''')

    conn.commit()
    conn.close()

def search_memories(query, memory_type='user', limit=30, db_path='memories.db'):
    """
    Basic text search for memories using SQL LIKE queries.
    
    Args:
        query (str): Search term to find memories
        memory_type (str): Type of memories to search ('user' or 'public')
        limit (int): Maximum number of memories to return
        db_path (str): Path to the SQLite database
        
    Returns:
        list: List of memory dictionaries matching the search criteria
    """
    # Check if database exists
    if not os.path.exists(db_path):
        print(f"No database found at {db_path}")
        return []
    
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Search across multiple fields
        cursor.execute(
            """
            SELECT * FROM memories 
            WHERE type = ? AND (
                title LIKE ? OR 
                location LIKE ? OR 
                description LIKE ? OR
                keywords LIKE ?
            ) 
            ORDER BY weight DESC 
            LIMIT ?
            """, 
            (
                memory_type, 
                f"%{query}%", 
                f"%{query}%", 
                f"%{query}%", 
                f"%{query}%", 
                limit
            )
        )
        
        # Convert results to list of dictionaries
        memories = [dict(row) for row in cursor.fetchall()]
        
        # Parse JSON fields
        for memory in memories:
            memory['keywords'] = json.loads(memory['keywords']) if memory['keywords'] else []
        
        return memories
        
    except Exception as e:
        print(f"Error searching memories: {e}")
        return []
        
    finally:
        if 'conn' in locals():
            conn.close()

def get_memory_by_id(memory_id, db_path='memories.db'):
    """Get a memory by its ID."""
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM memories WHERE id = ?", (memory_id,))
        memory = cursor.fetchone()
        
        if memory:
            memory = dict(memory)
            memory['keywords'] = json.loads(memory['keywords']) if memory['keywords'] else []
            
            if 'resnet_embedding' in memory and memory['resnet_embedding']:
                try:
                    memory['embedding_vector'] = json.loads(memory['resnet_embedding'])
                except json.JSONDecodeError:
                    memory['embedding_vector'] = []
            else:
                memory['embedding_vector'] = []
                
            if 'detected_objects' in memory and memory['detected_objects']:
                try:
                    memory['detected_objects_list'] = json.loads(memory['detected_objects'])
                except json.JSONDecodeError:
                    memory['detected_objects_list'] = []
        
        return memory
        
    except Exception as e:
        print(f"Error fetching memory {memory_id}: {e}")
        return None
        
    finally:
        if 'conn' in locals():
            conn.close()

def update_memory_weight(memory_id, new_weight, db_path='memories.db'):
    """Update a memory's weight."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Update both weight fields for compatibility
        cursor.execute(
            "UPDATE memories SET weight = ?, impact_weight = ? WHERE id = ?",
            (new_weight, new_weight, memory_id)
        )
        
        conn.commit()
        updated = cursor.rowcount > 0
        
        return updated
        
    except Exception as e:
        print(f"Error updating memory weight: {e}")
        return False
        
    finally:
        if 'conn' in locals():
            conn.close()

def get_memories_by_type(memory_type, limit=100, db_path='memories.db'):
    """Get memories by type."""
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM memories WHERE type = ? LIMIT ?", (memory_type, limit))
        memories = [dict(row) for row in cursor.fetchall()]
        
        # Parse JSON fields
        for memory in memories:
            memory['keywords'] = json.loads(memory['keywords']) if memory['keywords'] else []
            
            if 'resnet_embedding' in memory and memory['resnet_embedding']:
                try:
                    memory['embedding_vector'] = json.loads(memory['resnet_embedding'])
                except json.JSONDecodeError:
                    memory['embedding_vector'] = []
            else:
                memory['embedding_vector'] = []
        
        return memories
        
    except Exception as e:
        print(f"Error fetching memories by type: {e}")
        return []
        
    finally:
        if 'conn' in locals():
            conn.close()