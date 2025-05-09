# app/api/memories.py
from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Optional, Dict, Any
import logging
import sqlite3
import json
from datetime import datetime

from app.core.session import get_session_manager
from app.services.memory_recommendation import MemoryRecommendationEngine
from app.services.synthetic_memory_generator import get_synthetic_memory_generator
from app.models import Memory, NarrativeResponse, WeightAdjustmentResponse

# Configure router
router = APIRouter()

# Initialize recommendation engine
recommendation_engine = MemoryRecommendationEngine()

@router.get("/search", response_model=List[Memory])
def search_memories_endpoint(
    session_id: str,
    query: str,
    memory_type: str = 'all',  # 'all', 'user', or 'public'
    limit: int = 30,
    sort_by: str = "weight"
):
    """
    Search memories based on query using both keyword matching and embedding similarity.
    
    Args:
        session_id: Unique session identifier
        query: Search term to find memories
        memory_type: Type of memories to search ('all', 'user', or 'public')
        limit: Maximum number of memories to return
        sort_by: How to sort results ('weight', 'date', or 'relevance')
    """
    session_manager = get_session_manager()
    paths = session_manager.get_session_paths(session_id)
    
    if not paths:
        raise HTTPException(status_code=404, detail="Session not found")
    
    db_path = paths["db_path"]
    
    try:
        # Get all memories of the specified types
        if memory_type == 'all':
            user_memories = get_memories_by_type('user', limit=100, db_path=db_path)
            public_memories = get_memories_by_type('public', limit=100, db_path=db_path)
            all_memories = user_memories + public_memories
        else:
            all_memories = get_memories_by_type(memory_type, limit=100, db_path=db_path)
        
        if not all_memories:
            return []
        
        # Use recommendation engine to search memories with embeddings
        memories = recommendation_engine.search_memories_by_embeddings(
            query, 
            all_memories, 
            limit
        )
        
        if not memories:
            # No results from embedding search, try direct database search
            memories = search_memories(query, memory_type, limit, db_path)
        
        # Apply additional sorting if needed
        if sort_by == "date" and memories:
            memories = sorted(
                memories, 
                key=lambda m: datetime.strptime(m.get('date', '1900-01-01'), '%Y-%m-%d'),
                reverse=True
            )
        elif sort_by == "weight" and memories:
            memories = sorted(
                memories, 
                key=lambda m: float(m.get('weight', 1.0)),
                reverse=True
            )
        
        # Attach URLs for images
        for memory in memories:
            if 'filename' in memory and memory['filename']:
                memory_type = memory.get('type', 'user')
                memory['image_url'] = f"/api/static/{session_id}/{memory_type}/{memory['filename']}"
        
        return memories[:limit]
    
    except Exception as e:
        logging.error(f"Error in search endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error searching memories: {str(e)}")

@router.get("/similar/{memory_id}", response_model=List[Memory])
def get_similar_memories(
    session_id: str,
    memory_id: int,
    limit: int = 5,
    memory_type: str = 'all',
    use_embedding: bool = True
):
    """
    Find memories similar to the specified memory.
    
    Args:
        session_id: Unique session identifier
        memory_id: ID of the source memory
        limit: Maximum number of similar memories to return
        memory_type: Type of memories to search ('all', 'user', or 'public')
        use_embedding: Whether to use embedding-based similarity
    """
    session_manager = get_session_manager()
    paths = session_manager.get_session_paths(session_id)
    
    if not paths:
        raise HTTPException(status_code=404, detail="Session not found")
    
    db_path = paths["db_path"]
    
    # Get the source memory
    source_memory = get_memory_by_id(memory_id, db_path)
    
    if not source_memory:
        raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found")
    
    # Get all memories of the specified types
    if memory_type == 'all':
        user_memories = get_memories_by_type('user', limit=100, db_path=db_path)
        public_memories = get_memories_by_type('public', limit=100, db_path=db_path)
        all_memories = user_memories + public_memories
    else:
        all_memories = get_memories_by_type(memory_type, limit=100, db_path=db_path)
    
    # Find similar memories using the recommendation engine
    similar_memories = recommendation_engine.find_similar_memories(
        source_memory, 
        all_memories, 
        top_n=limit
    )
    
    # Attach URLs for images
    for memory in similar_memories:
        if 'filename' in memory and memory['filename']:
            memory_type = memory.get('type', 'user')
            memory['image_url'] = f"/api/static/{session_id}/{memory_type}/{memory['filename']}"
    
    return similar_memories

@router.get("/narrative", response_model=NarrativeResponse)
def generate_narrative_endpoint(
    session_id: str,
    query: str, 
    memory_type: str = 'all',
    max_memories: int = 5,
    prioritize_weights: bool = True
):
    """
    Generate a synthetic narrative from memories matching the query.
    
    Args:
        session_id: Unique session identifier
        query: Search term to find memories
        memory_type: Type of memories to search ('all', 'user', or 'public')
        max_memories: Maximum number of memories to include in the narrative
        prioritize_weights: Whether to prioritize memory weight over chronology
    """
    session_manager = get_session_manager()
    paths = session_manager.get_session_paths(session_id)
    
    if not paths:
        raise HTTPException(status_code=404, detail="Session not found")
    
    db_path = paths["db_path"]
    
    try:
        # First search memories
        if memory_type == 'all':
            user_memories = get_memories_by_type('user', limit=100, db_path=db_path)
            public_memories = get_memories_by_type('public', limit=100, db_path=db_path)
            all_memories = user_memories + public_memories
        else:
            all_memories = get_memories_by_type(memory_type, limit=100, db_path=db_path)
        
        if not all_memories:
            return {
                "text": f"No memories found of type '{memory_type}'.",
                "keywords": [],
                "highlighted_terms": [],
                "source_memories": []
            }
        
        # Search memories using the recommendation engine
        matching_memories = recommendation_engine.search_memories_by_embeddings(
            query, 
            all_memories, 
            limit=max_memories*2
        )
        
        if not matching_memories:
            return {
                "text": f"No memories found related to '{query}'.",
                "keywords": [],
                "highlighted_terms": [],
                "source_memories": []
            }
        
        # Sort memories based on priority
        if prioritize_weights:
            # Sort by weight first
            selected_memories = sorted(
                matching_memories, 
                key=lambda m: float(m.get('weight', 1.0)),
                reverse=True
            )[:max_memories]
        else:
            # Sort by date
            selected_memories = sorted(
                matching_memories,
                key=lambda m: m.get('date', '1900-01-01')
            )[:max_memories]
            
        # Generate narrative
        memory_generator = get_synthetic_memory_generator()
        narrative_result = memory_generator.generate_memory_narrative(selected_memories, query)
        
        # Prepare keywords with types
        keywords = [
            {"text": kw, "type": "primary"} for kw in narrative_result.get('keywords', [])
        ]
        
        return {
            "text": narrative_result.get('text', ''),
            "keywords": keywords,
            "highlighted_terms": narrative_result.get('highlighted_terms', []),
            "source_memories": narrative_result.get('source_memories', [])
        }
        
    except Exception as e:
        logging.error(f"Error generating narrative: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating narrative: {str(e)}")

@router.post("/{memory_id}/adjust_weight", response_model=WeightAdjustmentResponse)
def adjust_memory_weight_endpoint(
    session_id: str,
    memory_id: int, 
    adjustment: float = Query(0.1, ge=-1.0, le=1.0)
):
    """
    Adjust the weight of a memory by the specified amount.
    
    Args:
        session_id: Unique session identifier
        memory_id: ID of the memory to adjust
        adjustment: Amount to adjust the weight (positive or negative)
    """
    session_manager = get_session_manager()
    paths = session_manager.get_session_paths(session_id)
    
    if not paths:
        raise HTTPException(status_code=404, detail="Session not found")
    
    db_path = paths["db_path"]
    
    try:
        # Get the current memory
        memory = get_memory_by_id(memory_id, db_path)
        
        if not memory:
            raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found")
        
        current_weight = float(memory.get('weight', 1.0))
        new_weight = current_weight + adjustment
        
        # Ensure weight stays within reasonable bounds
        new_weight = max(0.1, min(5.0, new_weight))
        
        # Update the weight
        success = update_memory_weight(memory_id, new_weight, db_path)
        
        if not success:
            raise HTTPException(status_code=500, detail=f"Failed to update memory weight")
        
        return {
            "status": "success", 
            "message": f"Memory {memory_id} weight adjusted",
            "previous_weight": current_weight,
            "new_weight": new_weight
        }
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@router.post("/{memory_id}/increase_weight", response_model=WeightAdjustmentResponse)
def increase_weight_memory(session_id: str, memory_id: int):
    """Increase a memory's weight by 0.1."""
    return adjust_memory_weight_endpoint(session_id, memory_id, 0.1)
    
@router.post("/{memory_id}/decrease_weight", response_model=WeightAdjustmentResponse)
def decrease_weight_memory(session_id: str, memory_id: int):
    """Decrease a memory's weight by 0.1."""
    return adjust_memory_weight_endpoint(session_id, memory_id, -0.1)

@router.get("/{memory_id}")
def get_memory_endpoint(session_id: str, memory_id: int):
    """
    Get a memory by its ID.
    
    Args:
        session_id: Unique session identifier
        memory_id: ID of the memory to fetch
    """
    session_manager = get_session_manager()
    paths = session_manager.get_session_paths(session_id)
    
    if not paths:
        raise HTTPException(status_code=404, detail="Session not found")
    
    db_path = paths["db_path"]
    
    memory = get_memory_by_id(memory_id, db_path)
    
    if not memory:
        raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found")
    
    # Attach URL for image
    if 'filename' in memory and memory['filename']:
        memory_type = memory.get('type', 'user')
        memory['image_url'] = f"/api/static/{session_id}/{memory_type}/{memory['filename']}"
    
    return memory

# Utility functions
def get_memory_by_id(memory_id, db_path):
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
        logging.error(f"Error fetching memory {memory_id}: {e}")
        return None
        
    finally:
        if 'conn' in locals():
            conn.close()

def update_memory_weight(memory_id, new_weight, db_path):
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
        logging.error(f"Error updating memory weight: {e}")
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
        
        if memory_type == 'all':
            cursor.execute("SELECT * FROM memories LIMIT ?", (limit,))
        else:
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
        logging.error(f"Error fetching memories by type: {e}")
        return []
        
    finally:
        if 'conn' in locals():
            conn.close()

def search_memories(query, memory_type='all', limit=30, db_path='memories.db'):
    """
    Basic text search for memories using SQL LIKE queries.
    
    Args:
        query: Search term to find memories
        memory_type: Type of memories to search ('all', 'user', or 'public')
        limit: Maximum number of memories to return
        db_path: Path to the SQLite database
        
    Returns:
        list: List of memory dictionaries matching the search criteria
    """
    # Check if database exists
    if not os.path.exists(db_path):
        logging.error(f"No database found at {db_path}")
        return []
    
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Search across multiple fields
        if memory_type == 'all':
            cursor.execute(
                """
                SELECT * FROM memories 
                WHERE (
                    title LIKE ? OR 
                    location LIKE ? OR 
                    description LIKE ? OR
                    keywords LIKE ?
                ) 
                ORDER BY weight DESC 
                LIMIT ?
                """, 
                (
                    f"%{query}%", 
                    f"%{query}%", 
                    f"%{query}%", 
                    f"%{query}%", 
                    limit
                )
            )
        else:
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
        logging.error(f"Error searching memories: {e}")
        return []
        
    finally:
        if 'conn' in locals():
            conn.close()