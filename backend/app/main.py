# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from app.database import search_memories, update_memory_weight
from app.services.memory_recommendation import MemoryRecommendationEngine
from fastapi.middleware.cors import CORSMiddleware

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
    memories = search_memories(query, memory_type, limit)
    return memories

@app.get("/memories/narrative")
def generate_narrative_endpoint(
query: str,
memory_type: str = 'user'
):
    """Generate a synthetic narrative."""
    # First search memories
    memories = search_memories(query, memory_type)
    # Generate narrative
    narrative = recommendation_engine.generate_synthetic_narrative(memories)

    return {"narrative": narrative}

@app.post("/memories/{memory_id}/interact")
def interact_with_memory(memory_id: int):
    """Interact with a memory (increases its weight)."""
    try:
        update_memory_weight(memory_id)
        return {"status": "success", "message": "Memory interaction recorded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))