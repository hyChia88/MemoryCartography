from fastapi import APIRouter, HTTPException
from typing import List, Optional
from ..models import Memory, MemoryCreate, StatusResponse
from ..database import get_memories, add_memory, seed_data

router = APIRouter(
    prefix="/memories",
    tags=["memories"],
)

@router.get("/", response_model=List[Memory])
async def read_memories(type: str = "user"):
    """Get all memories of a specific type (user or public)."""
    return get_memories(memory_type=type)

@router.post("/", response_model=Memory)
async def create_memory(memory: MemoryCreate):
    """Create a new memory."""
    memory_id = add_memory(
        title=memory.title,
        location=memory.location,
        date=memory.date,
        memory_type=memory.type,
        keywords=memory.keywords,
        content=memory.content
    )
    
    return {
        "id": memory_id,
        **memory.dict()
    }

@router.post("/seed", response_model=StatusResponse)
async def seed_sample_data():
    """Seed the database with sample data."""
    success = seed_data()
    
    if success:
        return {"status": "success", "message": "Sample data inserted successfully."}
    else:
        return {"status": "info", "message": "Data already exists. No changes made."}