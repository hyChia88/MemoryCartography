from fastapi import APIRouter, HTTPException
from ..models import GenerateRequest, GenerateResponse
from ..database import find_memories_by_location
from ..services.text_service import generate_narrative

router = APIRouter(
    prefix="/generate",
    tags=["generate"],
)

@router.post("/", response_model=GenerateResponse)
async def generate_memory_narrative(request: GenerateRequest):
    """Generate a narrative about a location based on memories."""
    # Find memories related to the location
    memories = find_memories_by_location(
        location=request.location,
        memory_type=request.database_type
    )
    
    # Generate narrative and keywords
    result = generate_narrative(
        memories=memories,
        location=request.location,
        database_type=request.database_type
    )
    
    return result