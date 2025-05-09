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
from app.core.session import get_session_manager
from app.api.upload import router as upload_router
from app.api.memories import router as memories_router
from app.api.session import router as session_router
from app.models import Memory, NarrativeResponse, WeightAdjustmentResponse


# Near the top of main.py
import os
import logging

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('logs/api.log', mode='a'),
        logging.StreamHandler()
    ]
)

# Initialize FastAPI app
app = FastAPI(
    title="Privacy-Focused Memory Cartography API", 
    description="Spatial memory retrieval and analysis system with session-based temporary storage"
)

# Enable CORS
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

# Include routers
app.include_router(session_router, prefix="/api/session", tags=["Session Management"])
app.include_router(upload_router, prefix="/api/upload", tags=["File Upload"])
app.include_router(memories_router, prefix="/api/memories", tags=["Memory Search"])

# Background task for cleaning up expired sessions
def cleanup_expired_sessions():
    """Background task to clean up expired sessions."""
    session_manager = get_session_manager()
    session_manager.cleanup_expired_sessions()

@app.on_event("startup")
async def startup_event():
    """Run when the application starts."""
    # Schedule initial cleanup
    logging.info("Starting Memory Cartography API")

@app.get("/")
def read_root():
    """Root endpoint with API information."""
    return {
        "message": "Welcome to the Privacy-Focused Memory Cartography API", 
        "version": "1.0.0",
        "description": "This API processes images temporarily and does not permanently store user photos."
    }

# Schedule regular cleanup
@app.get("/api/maintenance/cleanup", tags=["Maintenance"])
async def trigger_cleanup(background_tasks: BackgroundTasks):
    """Manually trigger session cleanup (admin only)."""
    background_tasks.add_task(cleanup_expired_sessions)
    return {"message": "Cleanup initiated"}

# Mount temporary directories as static files for each session
# This will be done dynamically when a session is created
# See app/api/session.py for implementation

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)