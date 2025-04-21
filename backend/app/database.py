# database.py
import sqlite3
import json
from datetime import datetime
from pathlib import Path
import os

def init_db(db_path):
    """Initialize the SQLite database."""
    # Ensure the directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
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

def search_memories(query, memory_type='user', limit=10, db_path='memories.db'):
    """
    Search memories by query with robust error handling and flexible searching.
    
    Args:
        query (str): Search term to find memories
        memory_type (str, optional): Type of memories to search. Defaults to 'user'.
        limit (int, optional): Maximum number of memories to return. Defaults to 10.
        db_path (str, optional): Path to the SQLite database. Defaults to 'memories.db'.
    
    Returns:
        list: List of memory dictionaries matching the search criteria
    
    Raises:
        FileNotFoundError: If the database file does not exist
        sqlite3.Error: For database-related errors
    """
    # Check if database exists
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"No existing database found at {db_path}")
    
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Flexible search across multiple fields
        cursor.execute(
            """
            SELECT * FROM memories 
            WHERE type = ? AND (
                LOWER(title) LIKE LOWER(?) OR 
                LOWER(location) LIKE LOWER(?) OR 
                LOWER(keywords) LIKE LOWER(?) OR
                LOWER(description) LIKE LOWER(?)
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

        # Safely deserialize keywords
        for memory in memories:
            try:
                memory['keywords'] = json.loads(memory['keywords']) if memory.get('keywords') else []
            except json.JSONDecodeError:
                memory['keywords'] = []

        return memories

    except sqlite3.Error as e:
        # Log the error for debugging
        print(f"Database error: {e}")
        raise

    finally:
        # Ensure connection is closed
        if conn:
            conn.close()

# This function should be added to database.py

def search_memories_with_embeddings(query, memory_type='user', limit=10, db_path='memories.db'):
    """
    Search memories using both keyword matching and embedding similarity.
    
    Args:
        query (str): Search term to find memories
        memory_type (str): Type of memories to search ('user' or 'public')
        limit (int): Maximum number of memories to return
        db_path (str): Path to the SQLite database
    
    Returns:
        list: List of memory dictionaries matching the search criteria
    """
    import os
    import json
    import sqlite3
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Check if database exists
    if not os.path.exists(db_path):
        print(f"No database found at {db_path}")
        return []
    
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # First, get all memories of the specified type
        cursor.execute("SELECT * FROM memories WHERE type = ?", (memory_type,))
        all_memories = [dict(row) for row in cursor.fetchall()]
        
        # If no memories found, return empty list
        if not all_memories:
            return []
        
        # Parse JSON fields
        for memory in all_memories:
            try:
                memory['keywords'] = json.loads(memory['keywords']) if memory['keywords'] else []
                memory['embedding_vector'] = json.loads(memory['embedding']) if memory['embedding'] else []
            except (json.JSONDecodeError, TypeError):
                memory['keywords'] = []
                memory['embedding_vector'] = []
        
        # Calculate relevance scores
        for memory in all_memories:
            # Initialize score
            score = 0.0
            
            # 1. Keyword matching (basic text search)
            query_lower = query.lower()
            title_match = query_lower in memory.get('title', '').lower()
            location_match = query_lower in memory.get('location', '').lower()
            desc_match = query_lower in memory.get('description', '').lower()
            
            # Check if query matches any keywords
            keyword_match = any(query_lower in kw.lower() for kw in memory.get('keywords', []))
            
            # Add score for exact matches
            if title_match:
                score += 5.0  # Title matches are highly relevant
            if location_match:
                score += 3.0  # Location matches are relevant
            if desc_match:
                score += 2.0  # Description matches are somewhat relevant
            if keyword_match:
                score += 4.0  # Keyword matches are very relevant
            
            # 2. Weight factor
            weight = memory.get('weight', 1.0)
            score *= weight  # Apply memory weight
            
            # Store the score
            memory['search_score'] = score
        
        # Sort by search score
        scored_memories = sorted(all_memories, key=lambda m: m.get('search_score', 0), reverse=True)
        
        # If we have fewer than limit/2 results with non-zero scores, use embeddings for semantic search
        if len([m for m in scored_memories if m.get('search_score', 0) > 0]) < limit/2:
            # Create a simple query embedding (in a real system, you would use a proper embedding model)
            import hashlib
            import struct
            
            # Create a simple embedding from the query
            md5 = hashlib.md5(query.encode('utf-8')).digest()
            query_embedding = [struct.unpack('f', md5[i:i+4])[0] for i in range(0, min(32, len(md5)), 4)]
            
            # Normalize the vector
            magnitude = sum(x**2 for x in query_embedding) ** 0.5
            if magnitude > 0:
                query_embedding = [x/magnitude for x in query_embedding]
            
            # Calculate similarity for memories with embeddings
            memories_with_embeddings = []
            for memory in all_memories:
                embedding = memory.get('embedding_vector', [])
                if embedding and len(embedding) == len(query_embedding):
                    # Calculate cosine similarity
                    dot_product = sum(a*b for a, b in zip(query_embedding, embedding))
                    memory['embedding_similarity'] = dot_product
                    memories_with_embeddings.append(memory)
            
            # Sort by embedding similarity
            semantic_memories = sorted(memories_with_embeddings, key=lambda m: m.get('embedding_similarity', 0), reverse=True)
            
            # Combine results: first exact matches, then semantic matches
            exact_matches = [m for m in scored_memories if m.get('search_score', 0) > 0]
            
            # Remove duplicates from semantic results
            exact_ids = [m['id'] for m in exact_matches]
            unique_semantic = [m for m in semantic_memories if m['id'] not in exact_ids]
            
            # Combine and limit results
            combined_results = exact_matches + unique_semantic
            return combined_results[:limit]
        else:
            # We have enough exact matches, just return those
            return scored_memories[:limit]
    
    except Exception as e:
        print(f"Error searching memories: {e}")
        return []
    finally:
        if conn:
            conn.close()
            

def get_memory_by_id(memory_id, db_path='memories.db'):
    """Get a memory by its ID."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM memories WHERE id = ?", (memory_id,))
    memory = cursor.fetchone()
    
    if memory:
        memory = dict(memory)
        memory['keywords'] = json.loads(memory['keywords']) if memory['keywords'] else []
    
    conn.close()
    return memory

def update_memory_weight(memory_id, new_weight, db_path='memories.db'):
    """Update a memory's weight."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute(
        "UPDATE memories SET weight = ? WHERE id = ?",
        (new_weight, memory_id)
    )
    
    conn.commit()
    updated = cursor.rowcount > 0
    conn.close()
    
    return updated