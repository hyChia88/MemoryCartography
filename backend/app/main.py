# backend/app/main.py
from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import os
from pathlib import Path

# Import routers
from .routers.memories import router as memories_router
from .database import init_db

# Create FastAPI app
app = FastAPI(
    title="Memory Cartography API",
    description="API for the Memory Cartography system",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*","http://localhost:3000"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Health check endpoint
@app.get("/", tags=["health"])
def read_root():
    """Root endpoint for API health check."""
    return {"status": "OK", "message": "Memory Cartography API is running"}

# Mount static files for images
# Make sure these directories exist first
user_images_dir = Path("data/processed/user_photos")
public_images_dir = Path("data/processed/public_photos")

if user_images_dir.exists():
    app.mount("/images/user", StaticFiles(directory=str(user_images_dir)), name="user_images")
else:
    print(f"WARNING: User images directory not found at {user_images_dir}")

if public_images_dir.exists():
    app.mount("/images/public", StaticFiles(directory=str(public_images_dir)), name="public_images")
else:
    print(f"WARNING: Public images directory not found at {public_images_dir}")

# Include routers
app.include_router(memories_router)

# Error handling
@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    """Handle all unhandled exceptions."""
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "type": type(exc).__name__}
    )

# Initialize database on startup
@app.on_event("startup")
def startup_event():
    """Initialize the database on startup."""
    try:
        db_path = 'data/metadata/memories.db'
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        init_db(db_path)
        print(f"Database initialized at {db_path}")
    except Exception as e:
        print(f"Error initializing database: {e}")

# Start the application with uvicorn if run directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)