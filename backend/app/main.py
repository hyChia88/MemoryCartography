# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import sqlite3
from app.database import search_memories
from app.services.memory_recommendation import MemoryRecommendationEngine
from fastapi.middleware.cors import CORSMiddleware
from app.services.synthetic_memory_generator import get_synthetic_memory_generator

app = FastAPI()
recommendation_engine = MemoryRecommendationEngine()

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

class Memory(BaseModel):
    id: int
    title: str
    location: str
    date: str
    keywords: List[str]
    type: str
    description: Optional[str] = None
    weight: float = 1.0

@app.get("/memories/search", response_model=List[Memory])
def search_memories_endpoint(
    query: str,
    memory_type: str = 'user',
    limit: int = 10
    ):
    """Search memories based on query."""
    memories = search_memories(query, memory_type, limit) # memories sort by weights
    return memories

@app.get("/memories/narrative")
def generate_narrative_endpoint(
    query: str, 
    memory_type: str = 'user'
):
    """Generate a synthetic narrative."""
    # First search memories
    memories = search_memories(query, memory_type)  # memories sort by weights
    
    # Use the recommendation_engine, sort memory by recency and importants
    selected_memoris_synt_sort = get_synthetic_memory_generator(memories)
    
    # Generate narrative
    narrative_result = selected_memoris_synt_sort.generate_memory_narrative(memories, query)
    
    # Prepare keywords with types
    keywords = [
        {"text": kw, "type": "primary"} for kw in narrative_result.get('keywords', [])
    ]
    
    return {
        "text": narrative_result.get('text', ''),
        "keywords": keywords,
        "highlighted_terms": narrative_result.get('highlighted_terms', [])
    }

@app.post("/memories/{memory_id}/increase_weight")
def increase_weight_memory(memory_id: int):
    """Interact with a memory (increases its weight)."""
    try:
        # Connect to the database to get current weight before update
        conn = sqlite3.connect('memories.db')
        cursor = conn.cursor()
        
        # Fetch current weight
        cursor.execute("SELECT weight FROM memories WHERE id = ?", (memory_id,))
        result = cursor.fetchone()
        
        if result is None:
            conn.close()
            raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found")
        
        current_weight = result[0] or 1.0
        new_weight = current_weight + 0.1
        
        # Update weight
        cursor.execute(
            "UPDATE memories SET weight = ? WHERE id = ?",
            (new_weight, memory_id)
        )
        
        conn.commit()
        conn.close()
        
        return {
            "status": "success", 
            "message": f"Memory {memory_id} interaction recorded",
            "previous_weight": current_weight,
            "new_weight": new_weight
        }
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    
@app.post("/memories/{memory_id}/decrease_weight")
def decrease_weigh_memory(memory_id: int):
    """Interact with a memory (increases its weight)."""
    try:
        # Connect to the database to get current weight before update
        conn = sqlite3.connect('memories.db')
        cursor = conn.cursor()
        
        # Fetch current weight
        cursor.execute("SELECT weight FROM memories WHERE id = ?", (memory_id,))
        result = cursor.fetchone()
        
        if result is None:
            conn.close()
            raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found")
        
        current_weight = result[0] or 1.0
        new_weight = current_weight - 0.1
        
        # Update weight
        cursor.execute(
            "UPDATE memories SET weight = ? WHERE id = ?",
            (new_weight, memory_id)
        )
        
        conn.commit()
        conn.close()
        
        return {
            "status": "success", 
            "message": f"Memory {memory_id} interaction recorded",
            "previous_weight": current_weight,
            "new_weight": new_weight
        }
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")