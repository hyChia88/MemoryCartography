# backend/app/routers/generate.py
from fastapi import APIRouter, HTTPException
from typing import Optional, List
from pydantic import BaseModel
from ..services.memory_recommendation import get_recommendation_engine

router = APIRouter(
    prefix="/generate",
    tags=["generate"],
)

class MemoryResponse(BaseModel):
    id: int
    filename: str
    title: str
    location: str
    date: str
    keywords: List[str]
    description: str
    image_path: str
    connection_keyword: Optional[str] = None

class KeywordItem(BaseModel):
    text: str
    type: str

class NarrativeResponse(BaseModel):
    text: str
    keywords: List[KeywordItem]
    primary_memories: List[MemoryResponse]
    connected_memories: List[MemoryResponse]

@router.get("/{memory_type}/{location}", response_model=NarrativeResponse)
async def generate_memory_narrative(memory_type: str, location: str):
    """Generate a memory narrative for a specific location."""
    if memory_type not in ["user", "public"]:
        raise HTTPException(status_code=400, detail="Memory type must be 'user' or 'public'")
    
    recommendation_engine = get_recommendation_engine()
    result = recommendation_engine.generate_memory_narrative(location, memory_type)
    
    return result