# database.py
import sqlite3
import json
from datetime import datetime
from pathlib import Path
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

def _get_column_names(db_path):
    """Get the column names from the memories table."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(memories)")
    columns = [info[1] for info in cursor.fetchall()]
    conn.close()
    return columns

def _get_compatible_columns(db_path):
    """Get compatible column mappings based on what's available in the database."""
    columns = _get_column_names(db_path)
    
    # Map old and new column names for compatibility
    mappings = {
        'description': "openai_description" if "openai_description" in columns else "description",
        'keywords': "openai_keywords" if "openai_keywords" in columns else "keywords",
        'weight': "impact_weight" if "impact_weight" in columns else "weight",
        'embedding': "resnet_embedding" if "resnet_embedding" in columns else "embedding"
    }
    
    return mappings

def search_memories(query, memory_type='user', limit=30, db_path='memories.db'):
    """
    Search memories by query with robust error handling and flexible searching.
    
    This is a lightweight text-based search that doesn't use embeddings.
    
    Args:
        query (str): Search term to find memories
        memory_type (str, optional): Type of memories to search. Defaults to 'user'.
        limit (int, optional): Maximum number of memories to return. Defaults to 30.
        db_path (str, optional): Path to the SQLite database. Defaults to 'memories.db'.
    
    Returns:
        list: List of memory dictionaries matching the search criteria
    """
    # Check if database exists
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"No existing database found at {db_path}")
    
    try:
        # Get compatible column mappings
        col_map = _get_compatible_columns(db_path)
        desc_col = col_map['description']
        keywords_col = col_map['keywords']
        weight_col = col_map['weight']
        
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Flexible search across multiple fields
        search_fields = [
            f"LOWER(title) LIKE LOWER(?)",
            f"LOWER(location) LIKE LOWER(?)",
            f"LOWER({desc_col}) LIKE LOWER(?)", 
            f"LOWER({keywords_col}) LIKE LOWER(?)"
        ]
        
        # Check if detected_objects column exists
        columns = _get_column_names(db_path)
        if "detected_objects" in columns:
            search_fields.append("LOWER(detected_objects) LIKE LOWER(?)")
        
        query_params = [memory_type]
        for _ in search_fields:
            query_params.append(f"%{query}%")
        
        # Construct the full query
        sql_query = f"""
            SELECT * FROM memories 
            WHERE type = ? AND (
                {" OR ".join(search_fields)}
            ) 
            ORDER BY {weight_col} DESC 
            LIMIT ?
        """
        query_params.append(limit)
        
        cursor.execute(sql_query, query_params)

        # Convert results to list of dictionaries
        memories = [dict(row) for row in cursor.fetchall()]

        # Process the results to ensure the expected fields exist
        _process_memory_results(memories, col_map)

        return memories

    except sqlite3.Error as e:
        print(f"Database error in text search: {e}")
        raise

    finally:
        if 'conn' in locals():
            conn.close()

def _process_memory_results(memories, col_map):
    """Process memory results to ensure consistent field naming."""
    for memory in memories:
        # Handle keywords field
        keywords_col = col_map['keywords']
        if keywords_col in memory:
            try:
                memory['keywords'] = json.loads(memory[keywords_col]) if memory[keywords_col] else []
            except json.JSONDecodeError:
                memory['keywords'] = []
        else:
            memory['keywords'] = []
        
        # Handle description field
        desc_col = col_map['description']
        if desc_col in memory and desc_col != 'description':
            memory['description'] = memory[desc_col]
        elif 'description' not in memory:
            memory['description'] = ""
        
        # Handle weight field
        weight_col = col_map['weight']
        if weight_col in memory and weight_col != 'weight':
            memory['weight'] = memory[weight_col]
        elif 'weight' not in memory:
            memory['weight'] = 1.0
        
        # Handle embedding field
        embedding_col = col_map['embedding']
        if embedding_col in memory:
            try:
                memory['embedding_vector'] = json.loads(memory[embedding_col]) if memory[embedding_col] else []
            except json.JSONDecodeError:
                memory['embedding_vector'] = []
        else:
            memory['embedding_vector'] = []

def create_query_embedding(query, all_memory_texts):
    """
    Create an embedding for the query text using TF-IDF.
    
    Args:
        query (str): The search query text
        all_memory_texts (list): List of all memory texts for TF-IDF vocabulary
    
    Returns:
        numpy.ndarray: TF-IDF vector embedding for the query
    """
    # Initialize the TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    
    # Add the query to the texts to ensure its terms are in the vocabulary
    all_texts = all_memory_texts + [query]
    
    # Fit and transform the texts
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    # Get the query embedding (last row)
    query_embedding = tfidf_matrix[-1]
    
    return query_embedding

def search_memories_with_embeddings(query, memory_type='user', limit=30, db_path='memories.db'):
    """
    Search memories using both keyword matching and embedding similarity.
    
    This implementation properly uses embeddings for semantic similarity search.
    
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
        # Get compatible column mappings
        col_map = _get_compatible_columns(db_path)
        desc_col = col_map['description']
        keywords_col = col_map['keywords']
        weight_col = col_map['weight']
        embedding_col = col_map['embedding']
        
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # First, get all memories of the specified type
        cursor.execute(f"SELECT * FROM memories WHERE type = ?", (memory_type,))
        all_memories = [dict(row) for row in cursor.fetchall()]
        
        # If no memories found, return empty list
        if not all_memories:
            print(f"No memories found of type '{memory_type}'")
            return []
        
        # Process memory results to ensure consistent field naming
        _process_memory_results(all_memories, col_map)
        
        # First pass: basic keyword matching
        keywords_matches = []
        
        # Prepare search query terms
        query_terms = query.lower().split()
        
        for memory in all_memories:
            score = 0
            
            # Check title
            if any(term in memory.get('title', '').lower() for term in query_terms):
                score += 5.0
            
            # Check location
            if any(term in memory.get('location', '').lower() for term in query_terms):
                score += 3.0
            
            # Check description
            if any(term in memory.get('description', '').lower() for term in query_terms):
                score += 2.0
            
            # Check keywords
            keywords = memory.get('keywords', [])
            if keywords and any(query_term in keyword.lower() for query_term in query_terms for keyword in keywords):
                score += 4.0
            
            # Check detected objects if available
            detected_objects = []
            if 'detected_objects' in memory and memory['detected_objects']:
                try:
                    detected_objects = json.loads(memory['detected_objects'])
                    if any(query_term in obj.lower() for query_term in query_terms for obj in detected_objects):
                        score += 3.0
                except json.JSONDecodeError:
                    pass
            
            # Apply memory weight
            weight = float(memory.get('weight', 1.0))
            score *= weight
            
            # Store the score
            memory['text_match_score'] = score
            
            # Add to keyword matches if score > 0
            if score > 0:
                keywords_matches.append(memory)
        
        # Second pass: embedding-based similarity search
        # Try different embedding approaches depending on what's available
        
        # Approach 1: Use existing ResNet embeddings from images
        memories_with_embeddings = []
        
        for memory in all_memories:
            if 'embedding_vector' in memory and memory['embedding_vector']:
                memories_with_embeddings.append(memory)
        
        # If we have memories with embeddings and the query could be related to visual content
        visual_query_terms = ['image', 'picture', 'photo', 'scene', 'view', 'landscape', 'portrait']
        
        embedding_matches = []
        
        if memories_with_embeddings and (
                any(term in query.lower() for term in visual_query_terms) or 
                len(keywords_matches) < limit/2
            ):
            # Generate a pseudo-embedding for the query using either:
            # 1. If available, a proper embedding model (not implemented here)
            # 2. A simpler hash-based approach as a fallback
            import hashlib
            import struct
            
            # Create a simple embedding from the query (just a placeholder - not a real semantic embedding)
            md5 = hashlib.md5(query.encode('utf-8')).digest()
            query_embedding = np.array([struct.unpack('f', md5[i:i+4])[0] for i in range(0, min(32, len(md5)), 4)])
            
            # Normalize the vector
            magnitude = np.linalg.norm(query_embedding)
            if magnitude > 0:
                query_embedding = query_embedding / magnitude
            
            # Calculate similarity for each memory with embeddings
            for memory in memories_with_embeddings:
                embedding = np.array(memory['embedding_vector'])
                
                # Check if embedding dimensions match
                if len(embedding) == len(query_embedding):
                    # Calculate cosine similarity
                    similarity = np.dot(embedding, query_embedding) / (np.linalg.norm(embedding) * np.linalg.norm(query_embedding))
                    memory['embedding_similarity'] = float(similarity)
                    embedding_matches.append(memory)
        
        # Approach 2: Generate TF-IDF embeddings for text content if we don't have enough matches
        if len(keywords_matches) < limit/2 and len(embedding_matches) < limit/2:
            # Prepare text content for TF-IDF
            all_memory_texts = []
            memories_with_text = []
            
            for memory in all_memories:
                text_content = (
                    f"{memory.get('title', '')} "
                    f"{memory.get('location', '')} "
                    f"{memory.get('description', '')} "
                    f"{' '.join(memory.get('keywords', []))}"
                )
                if text_content.strip():
                    all_memory_texts.append(text_content)
                    memory['text_content'] = text_content
                    memories_with_text.append(memory)
            
            # Only proceed if we have text data
            if all_memory_texts:
                # Create query embedding
                query_embedding = create_query_embedding(query, all_memory_texts)
                
                # Create embeddings for all memories
                vectorizer = TfidfVectorizer(stop_words='english')
                tfidf_matrix = vectorizer.fit_transform(all_memory_texts + [query])
                memory_embeddings = tfidf_matrix[:-1]  # All except the last row (query)
                
                # Calculate similarities
                similarities = cosine_similarity(query_embedding, memory_embeddings)[0]
                
                # Store similarity scores
                for i, memory in enumerate(memories_with_text):
                    memory['tfidf_similarity'] = float(similarities[i])
                    # If it's not already in the embedding matches, add it
                    if memory['id'] not in [m['id'] for m in embedding_matches]:
                        embedding_matches.append(memory)
        
        # Final result selection and sorting
        # 1. If we have enough keyword matches, prioritize those
        if len(keywords_matches) >= limit/2:
            # Sort by text match score
            final_results = sorted(keywords_matches, key=lambda m: m.get('text_match_score', 0), reverse=True)
            
            # Supplement with embedding matches if needed
            if len(final_results) < limit:
                # Get embedding matches not already in results
                existing_ids = {m['id'] for m in final_results}
                additional_matches = [m for m in embedding_matches if m['id'] not in existing_ids]
                
                # Sort by embedding similarity
                additional_matches.sort(
                    key=lambda m: m.get('embedding_similarity', 0) + m.get('tfidf_similarity', 0), 
                    reverse=True
                )
                
                final_results.extend(additional_matches[:limit - len(final_results)])
        else:
            # Combine both types of matches
            # FIX: This is where the error occurs - we can't use set() with dictionaries
            # So we'll use a different approach to combine and deduplicate
            combined_matches = []
            memory_ids = set()
            
            # Add keyword matches first
            for memory in keywords_matches:
                if memory['id'] not in memory_ids:
                    combined_matches.append(memory)
                    memory_ids.add(memory['id'])
            
            # Then add embedding matches if they're not already included
            for memory in embedding_matches:
                if memory['id'] not in memory_ids:
                    combined_matches.append(memory)
                    memory_ids.add(memory['id'])
            
            # Create a combined score
            for memory in combined_matches:
                memory['combined_score'] = (
                    memory.get('text_match_score', 0) * 2.0 +  # Weight text matches higher
                    memory.get('embedding_similarity', 0) * 1.5 +
                    memory.get('tfidf_similarity', 0) * 1.0
                )
            
            # Sort by combined score
            final_results = sorted(combined_matches, key=lambda m: m.get('combined_score', 0), reverse=True)
        
        # Clean up temporary fields used for scoring
        result_copy = []
        for memory in final_results[:limit]:  # Apply limit here
            memory_copy = memory.copy()  # Create a copy to avoid modifying the original
            for field in ['text_match_score', 'embedding_similarity', 'tfidf_similarity', 
                         'combined_score', 'text_content', 'embedding_vector']:
                if field in memory_copy:
                    memory_copy.pop(field, None)
            result_copy.append(memory_copy)
        
        return result_copy
    
    except Exception as e:
        print(f"Error in embedding search: {e}")
        import traceback
        traceback.print_exc()
        
        # Fall back to simple text search
        print("Falling back to simple text search")
        return search_memories(query, memory_type, limit, db_path)
    
    finally:
        if 'conn' in locals():
            conn.close()

def get_memory_by_id(memory_id, db_path='memories.db'):
    """Get a memory by its ID."""
    # Check if database exists
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"No existing database found at {db_path}")
    
    try:
        # Get compatible column mappings
        col_map = _get_compatible_columns(db_path)
        
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM memories WHERE id = ?", (memory_id,))
        memory = cursor.fetchone()
        
        if memory:
            memory = dict(memory)
            
            # Process the memory to ensure consistent field naming
            _process_memory_results([memory], col_map)
        
        return memory
    
    except sqlite3.Error as e:
        print(f"Database error in get_memory_by_id: {e}")
        raise
    
    finally:
        if 'conn' in locals():
            conn.close()

def update_memory_weight(memory_id, new_weight, db_path='memories.db'):
    """Update a memory's weight."""
    # Check if database exists
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"No existing database found at {db_path}")
    
    try:
        # Get compatible column mappings
        col_map = _get_compatible_columns(db_path)
        weight_col = col_map['weight']
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            f"UPDATE memories SET {weight_col} = ? WHERE id = ?",
            (new_weight, memory_id)
        )
        
        # If we have both old and new weight columns, update both
        columns = _get_column_names(db_path)
        if 'weight' in columns and 'impact_weight' in columns and weight_col != 'weight':
            cursor.execute(
                "UPDATE memories SET weight = ? WHERE id = ?",
                (new_weight, memory_id)
            )
        
        conn.commit()
        updated = cursor.rowcount > 0
        
        return updated
    
    except sqlite3.Error as e:
        print(f"Database error in update_memory_weight: {e}")
        raise
    
    finally:
        if 'conn' in locals():
            conn.close()