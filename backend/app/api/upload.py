from fastapi import APIRouter, HTTPException, File, UploadFile, Form, BackgroundTasks, Depends, Query
from pydantic import BaseModel  # Add this import for the ProcessRequest model
from typing import List, Optional
import os
import shutil
import logging
from pathlib import Path
import aiofiles
import asyncio

from app.core.session import get_session_manager
from app.services.process_data import DataProcessor
from app.services.web_scraper import LocationImageScraper

# Configure router
router = APIRouter()

# Setup logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('logs/upload.log', mode='a'),
        logging.StreamHandler()
    ]
)

# Create a model for the process request
class ProcessRequest(BaseModel):
    session_id: str

@router.post("/photos")
async def upload_photos(
    session_id: str = Query(...),  # Changed from Form to Query
    files: List[UploadFile] = File(...)
):
    """
    Upload user photos for processing in a temporary session.
    
    Args:
        session_id: Unique session identifier (as query parameter)
        files: List of image files to upload
        
    Returns:
        Dict with upload status information
    """
    # Log the request details for debugging
    logging.info(f"Upload request received for session {session_id} with {len(files)} files")
    
    # Get session paths
    session_manager = get_session_manager()
    paths = session_manager.get_session_paths(session_id)
    
    if not paths:
        logging.error(f"Session not found: {session_id}")
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Save uploaded files to raw directory
    raw_user_dir = paths["raw_user"]
    saved_files = []
    
    try:
        for file in files:
            if file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.gif')):
                # Create safe filename
                safe_filename = os.path.basename(file.filename)
                file_path = os.path.join(raw_user_dir, safe_filename)
                
                # Ensure the directory exists
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
                # Log file details
                logging.info(f"Processing file: {file.filename}, saving to {file_path}")
                
                try:
                    # Save the file
                    contents = await file.read()
                    with open(file_path, 'wb') as f:
                        f.write(contents)
                    
                    saved_files.append(file.filename)
                    logging.info(f"Saved file: {file_path}")
                except Exception as e:
                    logging.error(f"Error saving file {file.filename}: {e}")
    except Exception as e:
        logging.error(f"Error during file upload: {e}")
        raise HTTPException(status_code=500, detail=f"Error uploading files: {str(e)}")
    
    logging.info(f"Upload complete for session {session_id}: {len(saved_files)} files saved")
    
    return {
        "session_id": session_id,
        "uploaded_files": saved_files,
        "total_files": len(saved_files),
        "status": "success" if saved_files else "no_files_saved"
    }

@router.post("/process")
async def process_photos(
    request: ProcessRequest,
    background_tasks: BackgroundTasks,
    session_id: str = Query(...)
):
    """
    Process the uploaded photos in the session.
    
    Args:
        request: Request with session_id
        background_tasks: FastAPI background tasks
        
    Returns:
        Dict with processing status information
    """
    session_id = request.session_id
    
    # Log the request
    logging.info(f"Processing request received for session: {session_id}")
    
    # Get session paths
    session_manager = get_session_manager()
    paths = session_manager.get_session_paths(session_id)
    
    if not paths:
        logging.error(f"Session not found: {session_id}")
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        # Initialize the data processor
        processor = DataProcessor(
            raw_user_dir=paths["raw_user"],
            raw_public_dir=paths["raw_public"],
            processed_user_dir=paths["processed_user"],
            processed_public_dir=paths["processed_public"],
            metadata_dir=paths["metadata"]
        )
        
        # Start processing in the background
        background_tasks.add_task(process_session_photos, processor, session_id, paths)
        
        return {
            "session_id": session_id,
            "status": "processing_started",
            "message": "Photo processing has been started in the background"
        }
    except Exception as e:
        logging.error(f"Error processing files for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing photos: {str(e)}")

# Add the missing process_session_photos function
async def process_session_photos(processor, session_id, paths):
    """
    Background task to process uploaded photos in a session.
    
    Args:
        processor: DataProcessor instance
        session_id: Unique session identifier
        paths: Session paths
    """
    logging.info(f"Starting background processing for session: {session_id}")
    
    try:
        # Process user photos
        user_metadata = processor.process_images(
            Path(paths["raw_user"]),
            Path(paths["processed_user"]),
            "user"
        )
        
        # Extract locations from processed metadata
        session_manager = get_session_manager()
        for _, data in user_metadata.items():
            location = data.get('location')
            session_manager.add_location(session_id, location)
        
        # Save metadata to database
        processor.create_sqlite_database(
            paths["db_path"],
            user_metadata,
            {}  # Empty public metadata for now
        )
        
        logging.info(f"Completed processing user photos for session: {session_id}")
    except Exception as e:
        logging.error(f"Error in background processing for session {session_id}: {e}")

# Add status endpoint to check processing progress
@router.get("/status/{session_id}")
async def get_processing_status(session_id: str):
    """
    Get the current processing status for a session.
    
    Args:
        session_id: Unique session identifier
        
    Returns:
        Dict with processing status information
    """
    # Get session paths
    session_manager = get_session_manager()
    paths = session_manager.get_session_paths(session_id)
    
    if not paths:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Check if database exists
    if not os.path.exists(paths["db_path"]):
        return {
            "session_id": session_id,
            "status": "not_processed",
            "message": "Photos have not been processed yet"
        }
    
    # Connect to the database
    try:
        import sqlite3
        conn = sqlite3.connect(paths["db_path"])
        cursor = conn.cursor()
        
        # Get count of processed photos
        cursor.execute("SELECT type, COUNT(*) FROM memories GROUP BY type")
        counts = {row[0]: row[1] for row in cursor.fetchall()}
        
        user_count = counts.get("user", 0)
        public_count = counts.get("public", 0)
        
        # Get locations
        cursor.execute("SELECT DISTINCT location FROM memories")
        locations = [row[0].split(',')[0].strip() for row in cursor.fetchall() 
                    if row[0] and row[0].lower() != "unknown location"]
        
        status = "processing"
        if user_count > 0 and public_count > 0:
            status = "completed"
        elif user_count > 0:
            status = "user_processed"
        
        conn.close()
        
        return {
            "session_id": session_id,
            "status": status,
            "user_photos_processed": user_count,
            "public_photos_processed": public_count,
            "locations_detected": list(set(locations)),
            "message": f"Processed {user_count} user photos and {public_count} public photos"
        }
    except Exception as e:
        logging.error(f"Error checking processing status: {e}")
        return {
            "session_id": session_id,
            "status": "error",
            "message": f"Error checking processing status: {str(e)}"
        }