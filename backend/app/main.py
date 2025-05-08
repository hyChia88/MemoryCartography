# main.py
from fastapi import FastAPI, HTTPException, Query, File, UploadFile, Form, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import sqlite3
import os
import json
from datetime import datetime
from app.database import init_db, get_memory_by_id, get_memories_by_type, update_memory_weight
from app.services.memory_recommendation import MemoryRecommendationEngine
from fastapi.middleware.cors import CORSMiddleware
from app.services.synthetic_memory_generator import get_synthetic_memory_generator
from fastapi.staticfiles import StaticFiles

# Define database and data paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'data/metadata/memories.db')
USER_PHOTOS_PATH = os.path.join(BASE_DIR, 'data/processed/user_photos/')
PUBLIC_PHOTOS_PATH = os.path.join(BASE_DIR, 'data/processed/public_photos/')
TEMP_DIR = os.path.join(BASE_DIR, 'data/temp/')

# Create directories if they don't exist
os.makedirs(USER_PHOTOS_PATH, exist_ok=True)
os.makedirs(PUBLIC_PHOTOS_PATH, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="Memory Cartography API", 
    description="Spatial memory retrieval and analysis system"
)

# Initialize recommendation engine
recommendation_engine = MemoryRecommendationEngine()

# Initialize database if it doesn't exist
if not os.path.exists(DB_PATH):
    init_db(DB_PATH)

# Mount static files directories for serving images
app.mount("/user-photos", StaticFiles(directory=USER_PHOTOS_PATH), name="user-photos")
app.mount("/public-photos", StaticFiles(directory=PUBLIC_PHOTOS_PATH), name="public-photos")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React development server
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Pydantic models
class Memory(BaseModel):
    id: int
    title: str
    location: str
    date: str
    keywords: List[str]
    type: str
    description: Optional[str] = None
    weight: float = 1.0
    filename: Optional[str] = None

class KeywordType(BaseModel):
    text: str
    type: str

class NarrativeResponse(BaseModel):
    text: str
    keywords: List[KeywordType]
    highlighted_terms: List[str]
    source_memories: List[int]

class WeightAdjustmentResponse(BaseModel):
    status: str
    message: str
    previous_weight: float
    new_weight: float

# API Routes
@app.get("/")
def read_root():
    return {"message": "Welcome to the Memory Cartography API", "version": "1.0.0"}

@app.get("/memories/search", response_model=List[Memory])
def search_memories_endpoint(
    query: str,
    memory_type: str = 'user',
    limit: int = 30,
    sort_by: str = "weight",
    db_path: str = DB_PATH
):
    """
    Search memories based on query using both keyword matching and embedding similarity.
    
    This endpoint implements a spatial memory search pipeline:
    1. Get all memories of the specified type
    2. Search using embeddings-based similarity
    3. Sort results based on combined score of weight, date, and relevance
    
    Args:
        query: Search term to find memories
        memory_type: Type of memories to search ('user' or 'public')
        limit: Maximum number of memories to return
        sort_by: How to sort results ('weight', 'date', or 'relevance')
    """
    try:
        # Get all memories of the specified type
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
            print("fallback method")
            from app.database import search_memories
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
        
        return memories[:limit]
    
    except Exception as e:
        print(f"Error in search endpoint: {e}")
        return []

@app.get("/memories/similar/{memory_id}", response_model=List[Memory])
def get_similar_memories(
    memory_id: int,
    limit: int = 5,
    memory_type: str = 'user',
    use_embedding: bool = True,
    db_path: str = DB_PATH
):
    """
    Find memories similar to the specified memory.
    
    Args:
        memory_id: ID of the source memory
        limit: Maximum number of similar memories to return
        memory_type: Type of memories to search ('user' or 'public')
        use_embedding: Whether to use embedding-based similarity
    """
    # Get the source memory
    source_memory = get_memory_by_id(memory_id, db_path)
    
    if not source_memory:
        raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found")
    
    # Get all memories of the specified type
    all_memories = get_memories_by_type(memory_type, limit=100, db_path=db_path)
    
    # Find similar memories using the recommendation engine
    similar_memories = recommendation_engine.find_similar_memories(
        source_memory, 
        all_memories, 
        top_n=limit
    )
    
    return similar_memories

@app.get("/memories/narrative", response_model=NarrativeResponse)
def generate_narrative_endpoint(
    query: str, 
    memory_type: str = 'user',
    max_memories: int = 5,
    prioritize_weights: bool = True,
    db_path: str = DB_PATH
):
    """
    Generate a synthetic narrative from memories matching the query.
    
    Args:
        query: Search term to find memories
        memory_type: Type of memories to search ('user' or 'public')
        max_memories: Maximum number of memories to include in the narrative
        prioritize_weights: Whether to prioritize memory weight over chronology
    """
    try:
        # First search memories
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
        print(f"Error generating narrative: {e}")
        return {
            "text": f"An error occurred while generating the narrative: {str(e)}",
            "keywords": [],
            "highlighted_terms": [],
            "source_memories": []
        }

@app.post("/memories/{memory_id}/adjust_weight", response_model=WeightAdjustmentResponse)
def adjust_memory_weight_endpoint(
    memory_id: int, 
    adjustment: float = Query(0.1, ge=-1.0, le=1.0),
    db_path: str = DB_PATH
):
    """
    Adjust the weight of a memory by the specified amount.
    
    Args:
        memory_id: ID of the memory to adjust
        adjustment: Amount to adjust the weight (positive or negative)
    """
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

# Convenience endpoints for increasing/decreasing weight
@app.post("/memories/{memory_id}/increase_weight", response_model=WeightAdjustmentResponse)
def increase_weight_memory(memory_id: int, db_path: str = DB_PATH):
    """Increase a memory's weight by 0.1."""
    return adjust_memory_weight_endpoint(memory_id, 0.1, db_path)
    
@app.post("/memories/{memory_id}/decrease_weight", response_model=WeightAdjustmentResponse)
def decrease_weight_memory(memory_id: int, db_path: str = DB_PATH):
    """Decrease a memory's weight by 0.1."""
    return adjust_memory_weight_endpoint(memory_id, -0.1, db_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)