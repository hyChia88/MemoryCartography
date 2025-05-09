# app/models.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# Memory schema
class MemoryBase(BaseModel):
    """Base model for Memory data."""
    title: str
    location: str
    date: str
    keywords: Optional[List[str]] = None
    description: Optional[str] = None
    filename: Optional[str] = None
    weight: Optional[float] = 1.0

class MemoryCreate(MemoryBase):
    """Model for creating a new Memory."""
    type: str = "user"  # 'user' or 'public'

class Memory(MemoryBase):
    """Model for a complete Memory with ID and additional fields."""
    id: int
    type: str
    image_url: Optional[str] = None  # URL to access the image
    openai_keywords: Optional[List[str]] = None
    openai_description: Optional[str] = None
    impact_weight: Optional[float] = 1.0
    detected_objects: Optional[List[str]] = None

class KeywordType(BaseModel):
    """Model for a keyword with type."""
    text: str
    type: str  # 'primary', 'related', or 'connected'

class NarrativeResponse(BaseModel):
    """Model for a generated narrative response."""
    text: str
    keywords: List[KeywordType]
    highlighted_terms: List[str]
    source_memories: List[int]

class WeightAdjustmentResponse(BaseModel):
    """Model for a memory weight adjustment response."""
    status: str
    message: str
    previous_weight: float
    new_weight: float

# Session models
class SessionCreate(BaseModel):
    """Model for session creation request."""
    pass  # No required fields for session creation

class SessionStatus(BaseModel):
    """Model for session status response."""
    session_id: str
    user_memories: int
    public_memories: int
    total_memories: int
    locations: List[str]
    first_date: Optional[str] = None
    status: str  # 'active', 'processing', 'expired'

class SessionDelete(BaseModel):
    """Model for session deletion response."""
    session_id: str
    status: str
    message: str

# Upload models
class UploadResponse(BaseModel):
    """Model for file upload response."""
    session_id: str
    uploaded_files: List[str]
    total_files: int
    status: str

class ProcessingResponse(BaseModel):
    """Model for processing status response."""
    session_id: str
    status: str  # 'processing_started', 'processing', 'completed', 'error'
    message: str

# Public photo fetching models
class FetchPublicRequest(BaseModel):
    """Model for public photo fetching request."""
    session_id: str
    max_photos_per_location: Optional[int] = 10
    total_limit: Optional[int] = 100

class FetchPublicResponse(BaseModel):
    """Model for public photo fetching response."""
    session_id: str
    status: str  # 'fetching_started', 'completed', 'error'
    locations: List[str]
    message: str

# Status response model
class StatusResponse(BaseModel):
    """Generic status response model."""
    status: str
    message: str