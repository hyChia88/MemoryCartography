# backend/app/routers/memories.py
from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from pydantic import BaseModel
from ..services.memory_recommendation import MemoryRecommendationEngine
from ..services.synthetic_memory_generator import get_synthetic_memory_generator
from ..database import (
    get_memories, 
    add_memory, 
    get_memory_by_id, 
    update_memory_weight, 
    record_memory_interaction,
    find_memories_by_location,
    seed_data
)

router = APIRouter(
    prefix="/api/memories",
    tags=["memories"],
)

# Models
class KeywordItem(BaseModel):
    text: str
    type: str  # 'primary' or 'connected'

class MemoryBase(BaseModel):
    title: str
    location: str
    date: str
    keywords: Optional[List[str]] = None
    description: Optional[str] = None
    weight: Optional[float] = 1.0

class MemoryCreate(MemoryBase):
    filename: str
    type: str = "user"  # 'user' or 'public'

class Memory(MemoryBase):
    id: int
    filename: str
    type: str
    image_path: str

class NarrativeResponse(BaseModel):
    text: str
    keywords: List[KeywordItem]
    highlighted_terms: List[str]
    primary_memories: List[Memory]
    connected_memories: List[Memory]

class StatusResponse(BaseModel):
    status: str
    message: str

# Initialize services
recommendation_engine = MemoryRecommendationEngine()
synthetic_memory_generator = get_synthetic_memory_generator()

# Routes
@router.get("/", response_model=List[Memory])
async def read_memories(type: str = "user"):
    """Get all memories of a specific type (user or public)."""
    memories = get_memories(memory_type=type)
    
    # Add image paths
    for memory in memories:
        if type == "user":
            memory["image_path"] = f"/images/user/{memory['filename']}"
        else:
            memory["image_path"] = f"/images/public/{memory['filename']}"
    
    return memories

@router.get("/{memory_id}", response_model=Memory)
async def read_memory(memory_id: int):
    """Get a specific memory by ID."""
    memory = get_memory_by_id(memory_id)
    
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")
    
    # Format image path for frontend
    if memory["type"] == "user":
        memory["image_path"] = f"/images/user/{memory['filename']}"
    else:
        memory["image_path"] = f"/images/public/{memory['filename']}"
    
    # Record view interaction
    record_memory_interaction(memory_id, "view")
    
    return memory

@router.post("/", response_model=Memory)
async def create_memory(memory: MemoryCreate):
    """Create a new memory."""
    memory_id = add_memory(
        filename=memory.filename,
        title=memory.title,
        location=memory.location,
        date=memory.date,
        memory_type=memory.type,
        keywords=memory.keywords,
        description=memory.description,
        weight=memory.weight
    )
    
    # Generate image path
    image_path = f"/images/{memory.type}/{memory.filename}"
    
    return {
        "id": memory_id,
        **memory.dict(),
        "image_path": image_path
    }

@router.post("/seed", response_model=StatusResponse)
async def seed_sample_data():
    """Seed the database with sample data."""
    success = seed_data()
    
    if success:
        return {"status": "success", "message": "Sample data inserted successfully."}
    else:
        return {"status": "info", "message": "Data already exists. No changes made."}

@router.post("/{memory_id}/weight", response_model=StatusResponse)
async def update_weight(memory_id: int, weight: Optional[float] = None):
    """Update the weight of a memory."""
    success = update_memory_weight(memory_id, new_weight=weight)
    
    if success:
        return {"status": "success", "message": "Memory weight updated."}
    else:
        raise HTTPException(status_code=404, detail="Memory not found")

@router.post("/{memory_id}/interact", response_model=StatusResponse)
async def record_interaction(memory_id: int, interaction_type: str):
    """Record a user interaction with a memory."""
    allowed_types = ["click", "view", "select", "favorite", "share"]
    
    if interaction_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Interaction type must be one of: {', '.join(allowed_types)}")
    
    success = record_memory_interaction(memory_id, interaction_type)
    
    if success:
        return {"status": "success", "message": f"Interaction '{interaction_type}' recorded."}
    else:
        raise HTTPException(status_code=404, detail="Memory not found")

@router.get("/search/{memory_type}", response_model=List[Memory])
async def search_memories(
    memory_type: str,
    query: str = Query(..., description="Search query"),
    limit: int = Query(10, description="Maximum number of results to return")
):
    """Search for memories based on keywords, location, etc."""
    if memory_type not in ["user", "public"]:
        raise HTTPException(status_code=400, detail="Memory type must be 'user' or 'public'")
    
    memories = recommendation_engine.search_memories(query, memory_type, limit)
    
    return memories

@router.get("/narrative/{memory_type}", response_model=NarrativeResponse)
async def generate_narrative(
    memory_type: str,
    query: str = Query(..., description="Search query"),
    min_weight: float = Query(3.0, description="Minimum total weight for synthetic memory"),
    max_memories: int = Query(10, description="Maximum number of memories to use")
):
    """Generate a memory narrative based on a search query."""
    if memory_type not in ["user", "public"]:
        raise HTTPException(status_code=400, detail="Memory type must be 'user' or 'public'")
    
    # Step 1: Find relevant memories
    relevant_memories = recommendation_engine.search_memories(
        query, memory_type=memory_type, limit=max_memories
    )
    
    if not relevant_memories:
        # Return empty response if no memories found
        return {
            "text": f"No memories found for '{query}'.",
            "keywords": [{"text": query, "type": "primary"}],
            "highlighted_terms": [],
            "primary_memories": [],
            "connected_memories": []
        }
    
    # Step 2: Generate synthetic memory
    synthetic_memory = synthetic_memory_generator.generate_memory_narrative(
        relevant_memories, query
    )
    
    # Step 3: Find connected memories
    connected_memories = []
    for memory in relevant_memories[:3]:  # Only use top 3 memories
        similar = recommendation_engine.find_similar_memories(
            memory['id'], memory_type=memory_type, limit=3
        )
        connected_memories.extend(similar)
    
    # Remove duplicates and memories already in primary list
    primary_ids = {m['id'] for m in relevant_memories}
    unique_connected = []
    seen_ids = set()
    
    for memory in connected_memories:
        if memory['id'] not in primary_ids and memory['id'] not in seen_ids:
            seen_ids.add(memory['id'])
            unique_connected.append(memory)
    
    # Step 4: Format keywords for response
    primary_keywords = []
    for kw in synthetic_memory.get('keywords', [])[:5]:
        primary_keywords.append({"text": kw, "type": "primary"})
    
    # Add search query as primary keyword if not already included
    query_keywords = query.split()
    for qk in query_keywords:
        if qk.lower() not in [k["text"].lower() for k in primary_keywords]:
            primary_keywords.append({"text": qk, "type": "primary"})
    
    # Add connected keywords from related memories
    connected_keywords = []
    for memory in unique_connected[:3]:
        for kw in memory.get('keywords', [])[:2]:
            if kw.lower() not in [k["text"].lower() for k in primary_keywords] and \
               kw.lower() not in [k["text"].lower() for k in connected_keywords]:
                connected_keywords.append({"text": kw, "type": "connected"})
    
    # Limit connected keywords
    connected_keywords = connected_keywords[:5]
    
    # Step 5: Construct final response
    response = {
        "text": synthetic_memory.get('text', ''),
        "keywords": primary_keywords + connected_keywords,
        "highlighted_terms": synthetic_memory.get('highlighted_terms', []),
        "primary_memories": relevant_memories,
        "connected_memories": unique_connected[:5]  # Limit to 5 connected memories
    }
    
    return response

@router.get("/similar/{memory_id}", response_model=List[Memory])
async def find_similar_memories(
    memory_id: int,
    memory_type: str = Query(..., description="Memory type (user or public)"),
    limit: int = Query(5, description="Maximum number of similar memories to return")
):
    """Find memories similar to a given memory."""
    if memory_type not in ["user", "public"]:
        raise HTTPException(status_code=400, detail="Memory type must be 'user' or 'public'")
    
    similar_memories = recommendation_engine.find_similar_memories(
        memory_id, memory_type=memory_type, limit=limit
    )
    
    if not similar_memories:
        raise HTTPException(status_code=404, detail="No similar memories found")
    
    return similar_memories

@router.get("/location/{memory_type}/{location}", response_model=List[Memory])
async def find_by_location(
    memory_type: str,
    location: str,
    limit: int = Query(10, description="Maximum number of memories to return")
):
    """Find memories by location."""
    if memory_type not in ["user", "public"]:
        raise HTTPException(status_code=400, detail="Memory type must be 'user' or 'public'")
    
    memories = find_memories_by_location(location, memory_type=memory_type, limit=limit)
    
    return memories