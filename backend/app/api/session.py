# app/api/session.py
from fastapi import APIRouter, HTTPException, Depends, Request
from typing import Dict, List, Any
import os
import logging
from datetime import datetime
from app.core.session import get_session_manager

# Configure router
router = APIRouter()

@router.post("/create")
async def create_session():
    """
    Create a new user session for temporary photo processing.
    
    Returns:
        Dict with session_id and creation timestamp
    """
    session_manager = get_session_manager()
    session_id = session_manager.create_session()
    
    return {
        "session_id": session_id,
        "created_at": datetime.now().isoformat(),
        "message": "Session created successfully. All data will be temporary."
    }

@router.get("/{session_id}/status")
async def get_session_status(session_id: str):
    """
    Get the status of a session including detected locations.
    
    Args:
        session_id: Unique session identifier
        
    Returns:
        Dict with session status information
    """
    session_manager = get_session_manager()
    paths = session_manager.get_session_paths(session_id)
    
    if not paths:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Connect to the session database
    db_path = paths["db_path"]
    try:
        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get memory count by type
        cursor.execute("SELECT type, COUNT(*) FROM memories GROUP BY type")
        memory_counts = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Get total memory count
        total_memories = sum(memory_counts.values())
        
        # Get locations
        cursor.execute("SELECT DISTINCT location FROM memories")
        locations = [row[0].split(',')[0].strip() for row in cursor.fetchall() if row[0] != "Unknown Location"]
        
        # Get session creation time
        cursor.execute("SELECT MIN(date) FROM memories")
        first_date = cursor.fetchone()[0]
        
        conn.close()
        
        # Get locations from session manager (which may include ones from images not yet in the database)
        all_locations = session_manager.get_locations(session_id)
        
        return {
            "session_id": session_id,
            "user_memories": memory_counts.get("user", 0),
            "public_memories": memory_counts.get("public", 0),
            "total_memories": total_memories,
            "locations": list(set(locations + all_locations)),
            "first_date": first_date,
            "status": "active"
        }
    except Exception as e:
        logging.error(f"Error getting session status: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting session status: {str(e)}")

@router.delete("/{session_id}")
async def delete_session(session_id: str):
    """
    Delete a session and all associated data.
    
    Args:
        session_id: Unique session identifier
        
    Returns:
        Dict with deletion status
    """
    session_manager = get_session_manager()
    success = session_manager.delete_session(session_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Session not found or could not be deleted")
    
    return {
        "session_id": session_id,
        "status": "deleted",
        "message": "Session and all associated data have been deleted."
    }

@router.get("/{session_id}/locations")
async def get_session_locations(session_id: str):
    """
    Get all detected locations in a session.
    
    Args:
        session_id: Unique session identifier
        
    Returns:
        Dict with list of locations
    """
    session_manager = get_session_manager()
    
    if not session_manager.get_session_paths(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    
    locations = session_manager.get_locations(session_id)
    
    return {
        "session_id": session_id,
        "locations": locations
    }

# Route to expose static files for a session
@router.get("/{session_id}/mount-static")
async def mount_static_files(session_id: str, request: Request):
    """
    Mount static file directories for a session.
    This is an admin endpoint that should be called after session creation.
    
    Args:
        session_id: Unique session identifier
        
    Returns:
        Dict with mount status
    """
    session_manager = get_session_manager()
    paths = session_manager.get_session_paths(session_id)
    
    if not paths:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # In a production app, you'd dynamically mount directories
    # For FastAPI, this is typically done at startup, not runtime
    # This is a placeholder that shows the intent
    
    return {
        "session_id": session_id,
        "message": "Static files would be mounted here in a production setup",
        "user_photos_url": f"/static/{session_id}/user",
        "public_photos_url": f"/static/{session_id}/public"
    }