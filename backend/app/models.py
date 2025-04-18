from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# Memory schema
class MemoryBase(BaseModel):
    title: str
    location: str
    date: str
    keywords: Optional[List[str]] = None
    content: Optional[str] = None

class MemoryCreate(MemoryBase):
    type: str = "user"  # 'user' or 'public'

class Memory(MemoryBase):
    id: int
    type: str

# Request models
class GenerateRequest(BaseModel):
    location: str
    database_type: str = "user"  # 'user' or 'public'

# Response models
class KeywordResponse(BaseModel):
    text: str
    type: str  # 'primary', 'related', or 'connected'

class GenerateResponse(BaseModel):
    text: str
    keywords: List[KeywordResponse]

# For a simple response
class StatusResponse(BaseModel):
    status: str
    message: str