# Standard Library Imports
import os
import json
import sqlite3
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple

# Third-Party Imports
from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer # For text query embedding
from openai import OpenAI
from dotenv import load_dotenv

# --- Application-Specific Imports ---
# Adjust these paths based on your project structure
try:
    from app.core.session import get_session_manager, SessionManager
except ImportError:
    logging.error("Failed to import session manager from app.core.session. Using dummy.")
    # Dummy Session Manager for structure - Replace with your actual implementation
    class SessionManager:
        def get_session_paths(self, session_id: str) -> Optional[Dict[str, str]]: return None
        def add_location(self, session_id: str, location: str): pass # Not used here but part of dummy
    _dummy_session_manager = SessionManager()
    def get_session_manager(): return _dummy_session_manager

# --- Configuration ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
router = APIRouter()

# --- Initialize Models ---
# Load a sentence transformer model for encoding text queries
# This will download the model on first run if not cached
try:
    # Using a common, efficient model. Choose one appropriate for your needs.
    text_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("SentenceTransformer model 'all-MiniLM-L6-v2' loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load SentenceTransformer model: {e}", exc_info=True)
    text_embedding_model = None

# Initialize OpenAI client for narrative generation
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_client = None
if openai_api_key:
    try:
        openai_client = OpenAI(api_key=openai_api_key)
        logger.info("OpenAI client initialized successfully for narrative generation.")
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
else:
    logger.warning("OPENAI_API_KEY not found. Narrative generation will be disabled.")


# --- Pydantic Models ---
class Memory(BaseModel):
    """Represents a memory item returned by the API."""
    id: int
    filename: str
    original_path: Optional[str] = None
    processed_path: Optional[str] = None
    title: Optional[str] = None
    location: Optional[str] = None
    date: Optional[str] = None
    type: str # 'user' or 'public'
    openai_keywords: Optional[List[str]] = Field(default_factory=list)
    openai_description: Optional[str] = None
    impact_weight: Optional[float] = 1.0
    # Exclude embedding from default response unless specifically requested
    # resnet_embedding: Optional[List[float]] = None
    detected_objects: Optional[List[str]] = Field(default_factory=list)
    image_url: Optional[str] = None # URL to access the image via API
    # Add score for relevance if needed
    relevance_score: Optional[float] = None

class NarrativeResponse(BaseModel):
    """Response model for the narrative generation endpoint."""
    session_id: str
    narrative_text: str
    source_memory_ids: List[int] = Field(default_factory=list)
    query: str


# --- Helper Functions ---

def get_db_connection(db_path: str) -> Optional[sqlite3.Connection]:
    """Establishes a connection to the SQLite database."""
    if not Path(db_path).exists():
        logger.error(f"Database file not found at: {db_path}")
        return None
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row # Return rows as dictionary-like objects
        logger.debug(f"Database connection established: {db_path}")
        return conn
    except sqlite3.Error as e:
        logger.error(f"Error connecting to database {db_path}: {e}", exc_info=True)
        return None

def parse_memory_row(row: sqlite3.Row) -> Dict[str, Any]:
    """Parses a database row into a memory dictionary, handling JSON fields."""
    memory = dict(row)
    try:
        memory['openai_keywords'] = json.loads(memory.get('openai_keywords', '[]') or '[]')
    except (json.JSONDecodeError, TypeError):
        memory['openai_keywords'] = []
    try:
        memory['detected_objects'] = json.loads(memory.get('detected_objects', '[]') or '[]')
    except (json.JSONDecodeError, TypeError):
        memory['detected_objects'] = []
    try:
        # Load embedding separately if needed, but usually large
        embedding_str = memory.get('resnet_embedding')
        if embedding_str:
            memory['resnet_embedding_vector'] = json.loads(embedding_str)
        else:
            memory['resnet_embedding_vector'] = None
    except (json.JSONDecodeError, TypeError):
         memory['resnet_embedding_vector'] = None
    # Remove the raw embedding string from the default dict
    memory.pop('resnet_embedding', None)
    return memory

def fetch_candidate_memories(conn: sqlite3.Connection, memory_type: str = 'all') -> List[Dict[str, Any]]:
    """Fetches all memories of a specific type with necessary fields for ranking."""
    memories = []
    try:
        cursor = conn.cursor()
        sql = """
            SELECT id, filename, original_path, processed_path, title, location, date, type,
                   openai_keywords, openai_description, impact_weight, resnet_embedding, detected_objects
            FROM memories
        """
        params = []
        if memory_type != 'all':
            sql += " WHERE type = ?"
            params.append(memory_type)

        cursor.execute(sql, params)
        rows = cursor.fetchall()
        memories = [parse_memory_row(row) for row in rows]
        logger.info(f"Fetched {len(memories)} candidate memories of type '{memory_type}'.")
    except sqlite3.Error as e:
        logger.error(f"Error fetching candidate memories: {e}", exc_info=True)
    return memories

def calculate_keyword_score(memory: Dict[str, Any], query: str) -> float:
    """Calculates a simple keyword matching score."""
    score = 0.0
    query_lower = query.lower()
    # Check keywords (most important)
    if any(query_lower in kw.lower() for kw in memory.get('openai_keywords', [])):
        score += 0.5
    # Check description
    if query_lower in (memory.get('openai_description', '') or '').lower():
        score += 0.3
    # Check title
    if query_lower in (memory.get('title', '') or '').lower():
        score += 0.1
    # Check location
    if query_lower in (memory.get('location', '') or '').lower():
        score += 0.1
    # Check detected objects
    if any(query_lower in obj.lower() for obj in memory.get('detected_objects', [])):
        score += 0.2

    return min(score, 1.0) # Cap score at 1.0


# --- API Endpoints ---

@router.get("/search", response_model=List[Memory])
def search_memories_endpoint(
    session_id: str = Query(..., description="Unique session identifier."),
    query: str = Query(..., description="Search query text."),
    memory_type: str = Query('all', description="Type of memories: 'all', 'user', or 'public'."),
    limit: int = Query(30, description="Maximum number of results to return."),
    sort_by: str = Query("relevance", description="Sort order: 'relevance', 'weight', 'date'.")
):
    """
    Searches memories using a combination of text query embedding similarity
    and keyword matching. Ranks results based on combined relevance or other criteria.
    """
    logger.info(f"Search request received for session '{session_id}': query='{query}', type='{memory_type}', limit={limit}, sort_by='{sort_by}'")

    session_manager = get_session_manager()
    paths = session_manager.get_session_paths(session_id)
    if not paths or "metadata" not in paths: # Check for metadata dir specifically
        logger.error(f"Session '{session_id}' not found or metadata path missing.")
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found or invalid configuration.")

    # Define the expected database path
    db_path = Path(paths["metadata"]) / f"{session_id}_memories.db"

    conn = get_db_connection(str(db_path))
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed.")

    ranked_memories = []
    try:
        # 1. Fetch all candidate memories from DB
        candidate_memories = fetch_candidate_memories(conn, memory_type)
        if not candidate_memories:
            logger.info("No candidate memories found.")
            return [] # Return empty list if no memories exist

        # 2. Calculate Relevance Scores
        query_embedding = None
        if text_embedding_model:
            try:
                # Generate embedding for the text query
                query_embedding = text_embedding_model.encode([query])[0]
                logger.debug(f"Generated query embedding for '{query}'.")
            except Exception as e:
                logger.error(f"Failed to generate query embedding: {e}", exc_info=True)
                # Continue without embedding search if model fails

        # Score each memory
        scored_memories = []
        for mem in candidate_memories:
            embedding_score = 0.0
            keyword_score = 0.0
            combined_score = 0.0

            # Calculate embedding similarity score (if possible)
            mem_embedding = mem.get('resnet_embedding_vector')
            if query_embedding is not None and mem_embedding is not None:
                try:
                    # Ensure embeddings are numpy arrays of compatible shape
                    q_emb = np.array(query_embedding).reshape(1, -1)
                    m_emb = np.array(mem_embedding).reshape(1, -1)
                    # Check if embedding dimensions match (ResNet50 is 2048, MiniLM is 384)
                    # Direct comparison is problematic. A multi-modal model or mapping is ideal.
                    # For now, we *cannot* directly compare MiniLM text embedding with ResNet image embedding.
                    # We will rely more heavily on keyword search or assume a compatible embedding engine exists.
                    # Placeholder: If a compatible embedding system were used:
                    # similarity = cosine_similarity(q_emb, m_emb)[0][0]
                    # embedding_score = max(0, similarity) # Ensure score is non-negative
                    logger.warning(f"Direct comparison between text query embedding ({q_emb.shape}) and image embedding ({m_emb.shape}) for memory {mem['id']} is not meaningful with current models. Skipping embedding score.")
                    embedding_score = 0.0 # Set to 0 as direct comparison isn't valid here
                except Exception as e:
                    logger.error(f"Error calculating similarity for memory {mem['id']}: {e}")
                    embedding_score = 0.0
            elif query_embedding is not None and mem_embedding is None:
                 logger.warning(f"Memory {mem['id']} is missing ResNet embedding. Cannot calculate embedding score.")


            # Calculate keyword matching score
            keyword_score = calculate_keyword_score(mem, query)

            # Combine scores (e.g., weighted average - adjust weights as needed)
            # Giving more weight to keywords since embedding comparison is currently invalid
            embedding_weight = 0.1 # Low weight due to incompatibility
            keyword_weight = 0.9
            combined_score = (embedding_score * embedding_weight) + (keyword_score * keyword_weight)

            mem['relevance_score'] = combined_score
            scored_memories.append(mem)
            logger.debug(f"Memory ID {mem['id']}: EmbScore={embedding_score:.2f}, KeyScore={keyword_score:.2f}, Combined={combined_score:.2f}")


        # 3. Sort Memories
        if sort_by == "relevance":
            # Sort by the combined score (descending)
            ranked_memories = sorted(scored_memories, key=lambda m: m.get('relevance_score', 0.0), reverse=True)
            logger.info(f"Sorting {len(scored_memories)} memories by relevance.")
        elif sort_by == "weight":
            # Sort by impact_weight (descending)
            ranked_memories = sorted(scored_memories, key=lambda m: m.get('impact_weight', 1.0), reverse=True)
            logger.info(f"Sorting {len(scored_memories)} memories by impact_weight.")
        elif sort_by == "date":
            # Sort by date (descending - newest first)
            ranked_memories = sorted(
                scored_memories,
                key=lambda m: datetime.strptime(m.get('date', '1900-01-01'), '%Y-%m-%d'),
                reverse=True
            )
            logger.info(f"Sorting {len(scored_memories)} memories by date.")
        else:
            # Default to relevance if sort_by is invalid
            logger.warning(f"Invalid sort_by parameter '{sort_by}'. Defaulting to relevance.")
            ranked_memories = sorted(scored_memories, key=lambda m: m.get('relevance_score', 0.0), reverse=True)


        # 4. Apply Limit
        final_memories = ranked_memories[:limit]
        logger.info(f"Returning {len(final_memories)} memories after limit.")

        # 5. Add Image URLs and Format Response
        response_list = []
        # base_static_path = f"/api/static/{session_id}" # Define base path for static files
        base_static_path = f"/api/session/static/{session_id}"
        for mem in final_memories:
            # Construct image URL relative to how static files are served
            img_url = None
            processed_rel_path = mem.get('processed_path') # e.g., "processed/user/user_0001_annotated.jpg"
            if processed_rel_path:
                 # Need to map processed_path to the static serving structure
                 # Assuming static serving maps session_id/type/filename
                 parts = Path(processed_rel_path).parts
                 if len(parts) >= 2: # Should contain type and filename
                      img_type_dir = parts[-2] # e.g., 'user'
                      img_filename = parts[-1] # e.g., 'user_0001_annotated.jpg'
                      img_url = f"{base_static_path}/{img_type_dir}/{img_filename}"
                 else:
                      logger.warning(f"Could not determine type/filename from processed_path: {processed_rel_path}")


            # Create Memory object for the response (excludes embedding vector)
            response_mem = Memory(
                id=mem['id'],
                filename=mem['filename'],
                original_path=mem.get('original_path'),
                processed_path=mem.get('processed_path'),
                title=mem.get('title'),
                location=mem.get('location'),
                date=mem.get('date'),
                type=mem['type'],
                openai_keywords=mem.get('openai_keywords', []),
                openai_description=mem.get('openai_description'),
                impact_weight=mem.get('impact_weight', 1.0),
                detected_objects=mem.get('detected_objects', []),
                image_url=img_url,
                relevance_score=mem.get('relevance_score')
            )
            response_list.append(response_mem)

        return response_list

    except Exception as e:
        logger.error(f"Error during memory search for session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during search: {str(e)}")
    finally:
        if conn:
            conn.close()
            logger.debug("Database connection closed.")


@router.get("/narrative", response_model=NarrativeResponse)
def generate_narrative_endpoint(
    session_id: str = Query(..., description="Unique session identifier."),
    query: str = Query(..., description="Original query used to find relevant memories."),
    max_memories: int = Query(5, ge=1, le=10, description="Maximum number of top memories to use for the narrative."),
    sort_by: str = Query("relevance", description="Criteria used to select top memories: 'relevance', 'weight', 'date'.")
):
    """
    Generates a short narrative based on the most relevant memories found for a query.
    """
    logger.info(f"Narrative request received for session '{session_id}': query='{query}', max_memories={max_memories}, sort_by='{sort_by}'")

    if not openai_client:
        raise HTTPException(status_code=501, detail="Narrative generation unavailable: OpenAI client not configured.")

    # 1. Perform search to get top N relevant memories
    # We call the search logic internally. Limit might be slightly larger initially if needed.
    try:
        # Use the same search logic, limit to max_memories needed
        top_memories_full = search_memories_endpoint(
            session_id=session_id,
            query=query,
            memory_type='all', # Search all types for narrative context
            limit=max_memories,
            sort_by=sort_by
        )
    except HTTPException as e:
         # If search itself fails, re-raise the exception
         logger.error(f"Search failed during narrative generation request: {e.detail}")
         raise e
    except Exception as e:
         logger.error(f"Unexpected error during search for narrative: {e}", exc_info=True)
         raise HTTPException(status_code=500, detail="Failed to retrieve memories for narrative.")


    if not top_memories_full:
        logger.warning(f"No memories found for query '{query}' in session '{session_id}' to generate narrative.")
        # Return a default response indicating no memories found
        return NarrativeResponse(
            session_id=session_id,
            narrative_text=f"I couldn't find any memories related to '{query}' to create a narrative.",
            source_memory_ids=[],
            query=query
        )

    # Convert Pydantic models back to simple dicts for processing if needed, or access attributes directly
    top_memories = [mem.model_dump() for mem in top_memories_full] # Use model_dump for Pydantic v2

    # 2. Prepare context for OpenAI
    narrative_context = f"Original Query: {query}\n\nKey Memories:\n"
    source_ids = []
    for i, mem in enumerate(top_memories):
        source_ids.append(mem['id'])
        narrative_context += f"\nMemory {i+1} (ID: {mem['id']}, Date: {mem.get('date', 'N/A')}, Location: {mem.get('location', 'N/A')}, Weight: {mem.get('impact_weight', 1.0):.1f}):\n"
        narrative_context += f"- Keywords: {', '.join(mem.get('openai_keywords', []))}\n"
        narrative_context += f"- Description: {mem.get('openai_description', 'N/A')}\n"
        narrative_context += f"- Detected Objects: {', '.join(mem.get('detected_objects', []))}\n"

    # 3. Create OpenAI Prompt
    prompt = f"""
    You are a creative storyteller. Based on the following memories retrieved for the query '{query}', weave a short, engaging narrative (2-4 sentences). The narrative should connect the memories thematically if possible, reflecting the mood suggested by the keywords and descriptions. Focus on the essence of the memories provided.

    {narrative_context}

    Generate the narrative:
    """

    # 4. Call OpenAI API
    try:
        logger.debug("Sending request to OpenAI for narrative generation...")
        completion = openai_client.chat.completions.create(
            model="gpt-3.5-turbo", # Use a cost-effective model for short narratives
            messages=[
                {"role": "system", "content": "You are a creative storyteller."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150, # Limit response length
            temperature=0.7 # Balance creativity and coherence
        )
        narrative_text = completion.choices[0].message.content.strip()
        logger.info(f"Narrative generated successfully for query '{query}'.")

    except Exception as e:
        logger.error(f"OpenAI API call failed during narrative generation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to generate narrative due to AI service error.")

    # 5. Return Response
    return NarrativeResponse(
        session_id=session_id,
        narrative_text=narrative_text,
        source_memory_ids=source_ids,
        query=query
    )
    
@router.post("/{memory_id}/adjust_weight")
async def adjust_memory_weight(
    memory_id: int,
    session_id: str = Query(..., description="Session identifier"),
    adjustment: float = Query(..., description="Weight adjustment amount (can be positive or negative)")
):
    """
    Adjust the weight of a specific memory by a given amount.
    """
    logger.info(f"Adjusting weight for memory {memory_id} by {adjustment} in session {session_id}")
    
    session_manager = get_session_manager()
    paths = session_manager.get_session_paths(session_id)
    if not paths or "metadata" not in paths:
        logger.error(f"Session '{session_id}' not found")
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    # Define the database path
    db_path = Path(paths["metadata"]) / f"{session_id}_memories.db"
    
    if not db_path.exists():
        raise HTTPException(status_code=404, detail="No memories database found for session")
    
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # First check if memory exists
        cursor.execute("SELECT impact_weight FROM memories WHERE id = ?", (memory_id,))
        row = cursor.fetchone()
        
        if not row:
            raise HTTPException(status_code=404, detail=f"Memory with ID {memory_id} not found")
        
        current_weight = row[0] or 1.0
        new_weight = max(0.1, min(10.0, current_weight + adjustment))  # Clamp between 0.1 and 10.0
        
        # Update the weight
        cursor.execute(
            "UPDATE memories SET impact_weight = ? WHERE id = ?",
            (new_weight, memory_id)
        )
        conn.commit()
        
        logger.info(f"Updated memory {memory_id} weight from {current_weight} to {new_weight}")
        
        return {
            "memory_id": memory_id,
            "old_weight": current_weight,
            "new_weight": new_weight,
            "adjustment": adjustment
        }
        
    except sqlite3.Error as e:
        logger.error(f"Database error adjusting weight for memory {memory_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        logger.error(f"Error adjusting memory weight: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    finally:
        if conn:
            conn.close()


# Include this router in your main FastAPI application
# Example main.py:
# from fastapi import FastAPI
# from app.api import memories # Assuming this file is app/api/memories.py
#
# app = FastAPI()
# app.include_router(memories.router, prefix="/api/memories", tags=["memories"])
