from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import memories, generate
from .database import init_db

# Create FastAPI app
app = FastAPI(title="Memory Cartography API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(memories.router)
app.include_router(generate.router)

# Initialize database on startup
@app.on_event("startup")
def startup_event():
    init_db()

@app.get("/")
def read_root():
    return {"message": "Welcome to Memory Cartography API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)