# app/main.py
from fastapi import FastAPI, HTTPException, Query, File, UploadFile, Form, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import List, Optional, Dict, Any
import os
import logging
import shutil
from pathlib import Path
import sqlite3
import json
from datetime import datetime

# Import application modules
# It's good practice to handle potential ImportErrors if files are missing during development
try:
    from core.session import get_session_manager # Assuming get_session_manager is needed globally or by other parts
except ImportError:
    get_session_manager = None # Define a fallback or handle appropriately
    logging.warning("Could not import get_session_manager from core.session.")

try:
    from api.upload import router as upload_router
    logging.info("✅ Upload router imported successfully")
except Exception as e:
    upload_router = None
    logging.error(f"❌ Could not import upload_router: {e}")
    import traceback
    logging.error(f"Full traceback: {traceback.format_exc()}")
    
# try:
#     from api.upload import router as upload_router
# except ImportError:
#     upload_router = None
#     logging.warning("Could not import upload_router. Upload API will be unavailable.")

try:
    from api.memories import router as memories_router
    logging.info("✅ Upload router imported successfully")
except Exception as e:
    memories_router = None
    logging.error(f"❌ Could not import memories_router: {e}")
    import traceback
    logging.error(f"Full traceback: {traceback.format_exc()}")

try:
    from api.session import router as session_router
except ImportError:
    session_router = None
    logging.warning("Could not import session_router. Session API will be unavailable.")

# Removed app.models import as it's not directly used in main.py
# from app.models import Memory, NarrativeResponse, WeightAdjustmentResponse


# Near the top of main.py
# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s: %(message)s', # Added name to format
    handlers=[
        logging.FileHandler('logs/api.log', mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__) # Use a specific logger for main.py messages


# Initialize FastAPI app
app = FastAPI(
    title="Privacy-Focused Memory Cartography API",
    description="Spatial memory retrieval and analysis system with session-based temporary storage",
    version="1.0.0" # Added version
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React development server
        "http://127.0.0.1:3000",
        "https://memory-cartography.vercel.app",
        "https://*.vercel.app",  
        "http://localhost:3001",
        # "http://localhost:8000", # Usually not needed if frontend and backend are on different ports
        # "http://127.0.0.1:8000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Include routers - CORRECTED VERSION
if session_router:
    app.include_router(session_router, prefix="/api/session", tags=["Session Management"])
    logger.info("Session router included at /api/session.")
else:
    logger.error("Session router not loaded.")

if upload_router:
    app.include_router(upload_router, prefix="/api/upload", tags=["File Upload & Processing"])
    logger.info("Upload router included at /api/upload.")
else:
    logger.error("Upload router not loaded.")

if memories_router:
    # FIXED: Changed from "/memories" to "/api/memories" to match frontend expectations
    app.include_router(memories_router, prefix="/api/memories", tags=["Memory Search & Narrative"])
    logger.info("Memories router included at /api/memories.")
else:
    logger.error("Memories router not loaded.")


# Background task for cleaning up expired sessions
# This function needs to be defined or imported if get_session_manager is available
def cleanup_expired_sessions():
    """Background task to clean up expired sessions."""
    if get_session_manager:
        session_manager = get_session_manager()
        if session_manager:
            logger.info("Running cleanup of expired sessions...")
            session_manager.cleanup_expired_sessions()
        else:
            logger.warning("Session manager not available for cleanup task.")
    else:
        logger.warning("get_session_manager not available for cleanup task.")


@app.on_event("startup")
async def startup_event():
    """Run when the application starts."""
    # You could schedule initial cleanup or other startup tasks here
    logger.info("Memory Cartography API starting up...")
    # Example: Trigger an initial cleanup if desired, though usually done periodically
    # background_tasks = BackgroundTasks() # Need to instantiate it if used directly here
    # background_tasks.add_task(cleanup_expired_sessions)


@app.get("/")
def read_root():
    """Root endpoint with API information."""
    return {
        "message": "Welcome to the Privacy-Focused Memory Cartography API",
        "version": app.version, # Use the app version
        "description": app.description, # Use the app description
        "docs_url": "/docs", # Point to default docs
        "redoc_url": "/redoc" # Point to default ReDoc
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for Railway."""
    return {"status": "healthy", "time": datetime.now().isoformat()}

# Schedule regular cleanup - This endpoint allows manual triggering.
# For automatic periodic cleanup, you'd typically use a library like FastAPI-Scheduler or an external cron job.
@app.get("/api/maintenance/cleanup", tags=["Maintenance"])
async def trigger_cleanup(background_tasks: BackgroundTasks):
    """
    Manually trigger session cleanup.
    Consider adding authentication/authorization for such an endpoint in a real application.
    """
    logger.info("Manual session cleanup triggered via API.")
    background_tasks.add_task(cleanup_expired_sessions)
    return {"message": "Session cleanup task initiated in the background."}

# --- Static File Serving ---
# Define a root directory for static files related to sessions.
STATIC_SESSIONS_ROOT = "static_session_files" # Should be at the project root or a configurable path
os.makedirs(STATIC_SESSIONS_ROOT, exist_ok=True)

# Mount this directory. If image_url is /api/static_content/session_id/type/filename.jpg
app.mount("/api/static_content", StaticFiles(directory=STATIC_SESSIONS_ROOT), name="static_session_content")
logger.info(f"Static files from '{STATIC_SESSIONS_ROOT}' mounted to /api/static_content")


if __name__ == "__main__":
    import uvicorn
    import os
    import sys
    
    # Add the parent directory to Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    # Get port from environment or default to 8000
    port = int(os.environ.get("PORT", 8000))
    
    print(f"Starting server on port {port}")
    print(f"Environment PORT: {os.environ.get('PORT', 'Not set')}")
    print(f"Working directory: {os.getcwd()}")
    
    uvicorn.run(
        app,  # Use the app object directly, not the string
        host="0.0.0.0",
        port=port,
        reload=False
    )