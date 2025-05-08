# main.py
from fastapi import FastAPI, HTTPException, Query, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import sqlite3
import os
import json
import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from app.database import init_db
from app.services.memory_recommendation import MemoryRecommendationEngine
from fastapi.middleware.cors import CORSMiddleware
from app.services.synthetic_memory_generator import get_synthetic_memory_generator
from fastapi.staticfiles import StaticFiles
import base64
import shutil

# Define database and data paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'data/metadata/memories.db')
USER_PHOTOS_PATH = os.path.join(BASE_DIR, 'data/processed/user_photos/')
PUBLIC_PHOTOS_PATH = os.path.join(BASE_DIR, 'data/processed/public_photos/')
TEMP_DIR = os.path.join(BASE_DIR, 'data/temp/')
print(DB_PATH)

# Create directories if they don't exist
os.makedirs(USER_PHOTOS_PATH, exist_ok=True)
os.makedirs(PUBLIC_PHOTOS_PATH, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

app = FastAPI(title="Memory Cartography API", description="Spatial memory retrieval and analysis system")

# Initialize recommendation engine
recommendation_engine = MemoryRecommendationEngine()

# Initialize database if it doesn't exist
if not os.path.exists(DB_PATH):
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    init_db(DB_PATH)

# Mount static files directories
app.mount("/user-photos", StaticFiles(directory=USER_PHOTOS_PATH), name="user-photos")
app.mount("/public-photos", StaticFiles(directory=PUBLIC_PHOTOS_PATH), name="public-photos")

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

# Pydantic models
class Memory(BaseModel):
    id: int
    title: str
    location: str
    date: str
    keywords: List[str]
    type: str
    description: Optional[str] = None
    weight: float = 1.0
    filename: Optional[str] = None

class MemoryCreate(BaseModel):
    title: str
    location: str
    date: str
    description: Optional[str] = None
    keywords: List[str]
    type: str = "user"
    weight: float = 1.0

class KeywordType(BaseModel):
    text: str
    type: str

class NarrativeResponse(BaseModel):
    text: str
    keywords: List[KeywordType]
    highlighted_terms: List[str]
    source_memories: List[int]

class WeightAdjustmentResponse(BaseModel):
    status: str
    message: str
    previous_weight: float
    new_weight: float

class ObjectDetectionResult(BaseModel):
    label: str
    confidence: float
    count: int

# Helper functions
def get_memory_by_id(memory_id: int, db_path: str = DB_PATH):
    """Helper function to get a memory by ID with error handling."""
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM memories WHERE id = ?", (memory_id,))
        memory = cursor.fetchone()
        
        if not memory:
            return None
            
        # Convert to dict and parse JSON fields
        memory_dict = dict(memory)
        memory_dict['keywords'] = json.loads(memory_dict['keywords']) if memory_dict['keywords'] else []
        
        # Parse embedding if present
        if 'embedding' in memory_dict and memory_dict['embedding']:
            try:
                memory_dict['embedding_vector'] = json.loads(memory_dict['embedding'])
            except json.JSONDecodeError:
                memory_dict['embedding_vector'] = []
        else:
            memory_dict['embedding_vector'] = []
            
        return memory_dict
    except Exception as e:
        print(f"Error fetching memory {memory_id}: {e}")
        return None
    finally:
        if conn:
            conn.close()

def get_memories_by_type(memory_type: str, limit: int = 100, db_path: str = DB_PATH):
    """Helper function to get memories by type with error handling."""
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM memories WHERE type = ? LIMIT ?", (memory_type, limit))
        memories = [dict(row) for row in cursor.fetchall()]
        
        # Parse JSON fields
        for memory in memories:
            memory['keywords'] = json.loads(memory['keywords']) if memory['keywords'] else []
            
            # Parse embedding if present
            if 'embedding' in memory and memory['embedding']:
                try:
                    memory_dict = memory
                    memory_dict['embedding_vector'] = json.loads(memory['embedding'])
                except json.JSONDecodeError:
                    memory_dict['embedding_vector'] = []
            else:
                memory['embedding_vector'] = []
                
        return memories
    except Exception as e:
        print(f"Error fetching memories of type {memory_type}: {e}")
        return []
    finally:
        if conn:
            conn.close()

def find_similar_memories_by_embedding(source_embedding, all_memories, exclude_id=None, top_n=5):
    """Find similar memories using vector embeddings."""
    if not all_memories or not source_embedding:
        return []
    
    # Extract embeddings and filter out memories without valid embeddings
    valid_memories = []
    memory_embeddings = []
    
    for memory in all_memories:
        # Skip the source memory if ID is provided
        if exclude_id and memory.get('id') == exclude_id:
            continue
            
        # Get embedding vector
        embedding = memory.get('embedding_vector', [])
        if not embedding or len(embedding) == 0:
            continue
            
        # Ensure embedding is the same dimension as source
        if len(embedding) != len(source_embedding):
            continue
            
        memory_embeddings.append(embedding)
        valid_memories.append(memory)
    
    if not valid_memories:
        return []
    
    # Calculate cosine similarities
    memory_embeddings = np.array(memory_embeddings)
    source_embedding = np.array(source_embedding).reshape(1, -1)
    similarities = cosine_similarity(source_embedding, memory_embeddings)[0]
    
    # Sort memories by similarity
    similar_indices = similarities.argsort()[::-1][:top_n]
    
    # Return the top similar memories
    return [valid_memories[i] for i in similar_indices]

# API Routes

@app.get("/")
def read_root():
    return {"message": "Welcome to the Memory Cartography API", "version": "1.0.0"}


@app.get("/memories/search", response_model=List[Memory])
def search_memories_endpoint(
    query: str,
    memory_type: str = 'user',
    limit: int = 30,
    sort_by: str = "weight", # Add sort parameter
    db_path=DB_PATH
    ):
    """
    Search memories based on query using both keyword matching and embedding similarity.
    Results can be sorted by weight (default), date, or relevance.
    """
    try:
        # Import the search function
        from app.database import search_memories_with_embeddings
        print("using advacn mtd")
        
        # Use the enhanced search function that leverages embeddings
        memories = find_similar_memories_by_embedding(query, memory_type, limit * 2, db_path=DB_PATH)
        
        if not memories:
            # Log the search attempt that returned no results
            print(f"Search for '{query}' in {memory_type} memories returned no results")
            
            # Fallback to a more lenient search if exact match fails
            # Try to search for partial matches in any field
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Use a more permissive LIKE query
            cursor.execute(
                """
                SELECT * FROM memories 
                WHERE type = ? AND (
                    title LIKE ? OR 
                    location LIKE ? OR 
                    description LIKE ? OR
                    keywords LIKE ?
                ) 
                LIMIT ?
                """, 
                (
                    memory_type, 
                    f"%{query}%", 
                    f"%{query}%", 
                    f"%{query}%", 
                    f"%{query}%", 
                    limit * 2
                )
            )
            
            fallback_memories = [dict(row) for row in cursor.fetchall()]
            conn.close()
            
            # Parse JSON fields
            for memory in fallback_memories:
                try:
                    memory['keywords'] = json.loads(memory['keywords']) if memory['keywords'] else []
                except (json.JSONDecodeError, TypeError):
                    memory['keywords'] = []
            
            # If we found results with the fallback, use those
            if fallback_memories:
                print(f"Fallback search found {len(fallback_memories)} results")
                memories = fallback_memories
        
        # Sort the memories based on the combined score of weight(0.3) + date(0.2) + relevance(0.5)
        def calculate_combined_score(memory):
            # Get weight score (normalized to 0-1 range)
            weight_score = float(memory.get('weight', 1.0)) / 5.0  # Assuming max weight is 5.0
            
            # Get date score (normalized to 0-1 range)
            try:
                date_obj = datetime.strptime(memory.get('date', '1900-01-01'), '%Y-%m-%d')
                # Convert to timestamp and normalize (assuming dates within last 10 years)
                max_date = datetime.now()
                min_date = max_date.replace(year=max_date.year - 10)
                date_score = (date_obj.timestamp() - min_date.timestamp()) / (max_date.timestamp() - min_date.timestamp())
            except:
                date_score = 0.0
            
            # Get relevance score (assuming it's stored in memory or calculated)
            relevance_score = float(memory.get('similarity', 0.0))  # Assuming similarity score is stored
            
            # Calculate combined score with weights
            combined_score = (weight_score * 0.3) + (date_score * 0.2) + (relevance_score * 0.5)
            return combined_score
        
        # Sort memories by combined score
        memories = sorted(memories, key=calculate_combined_score, reverse=True)
        
        # Return only the top 'limit' memories after sorting
        return memories[:limit]
    
    except Exception as e:
        print(f"Error in search endpoint: {e}")

@app.get("/memories/similar/{memory_id}", response_model=List[Memory])
def get_similar_memories(
    memory_id: int,
    limit: int = 5,
    memory_type: str = 'user',
    use_embedding: bool = True
):
    """Find memories similar to the specified memory."""
    # Get the source memory
    source_memory = get_memory_by_id(memory_id)
    
    if not source_memory:
        raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found")
    
    # Get all memories of the specified type
    all_memories = get_memories_by_type(memory_type)
    
    # Find similar memories based on method
    if use_embedding and source_memory.get('embedding_vector') and len(source_memory['embedding_vector']) > 0:
        # Use embedding-based similarity if available
        similar_memories = find_similar_memories_by_embedding(
            source_memory['embedding_vector'], 
            all_memories, 
            exclude_id=memory_id,
            top_n=limit
        )
    else:
        # Fallback to text-based similarity
        similar_memories = recommendation_engine.find_similar_memories(
            source_memory, 
            all_memories, 
            top_n=limit
        )
    
    return similar_memories

@app.get("/memories/narrative", response_model=NarrativeResponse)
def generate_narrative_endpoint(
    query: str, 
    memory_type: str = 'user',
    use_embedding: bool = True,
    max_memories: int = 30,
    prioritize_weights: bool = True
):
    """
    Generate a synthetic narrative from memories matching the query with improved weight prioritization.
    The prioritize_weights parameter determines if weights should be emphasized over chronology.
    """
    try:
        # First search memories - use the search endpoint with weight sorting
        from app.database import search_memories_with_embeddings
        
        # Get double the memories we need to have more options for selection
        all_memories = search_memories_with_embeddings(query, memory_type, limit=max_memories*2, db_path=DB_PATH)
        
        if not all_memories:
            # No memories found
            return {
                "text": f"No memories found related to '{query}'.",
                "keywords": [],
                "highlighted_terms": [],
                "source_memories": []
            }
        
        # Sort memories by weight first
        weight_sorted_memories = sorted(
            all_memories, 
            key=lambda m: float(m.get('weight', 1.0)),
            reverse=True
        )
        
        # Take a weighted sample of memories based on their importance
        # This ensures high-weight memories are more likely to be included
        selected_memories = []
        total_weight_target = 8.0  # Target combined weight
        current_total = 0.0
        
        # Always include the highest weighted memory
        if weight_sorted_memories:
            top_memory = weight_sorted_memories[0]
            selected_memories.append(top_memory)
            current_total += float(top_memory.get('weight', 1.0))
            
            # Then add more memories until we reach our target weight
            for memory in weight_sorted_memories[1:max_memories]:
                memory_weight = float(memory.get('weight', 1.0))
                
                # If this memory would put us over our target, we're done
                if current_total >= total_weight_target:
                    break
                
                # Add this memory
                selected_memories.append(memory)
                current_total += memory_weight

        # Ensure we have at least 2 memories if available (for more interesting narratives)
        if len(selected_memories) < 2 and len(weight_sorted_memories) >= 2:
            if len(selected_memories) == 0:
                selected_memories = weight_sorted_memories[:2]
            elif len(selected_memories) == 1 and len(weight_sorted_memories) > 1:
                # Add the next memory
                selected_memories.append(weight_sorted_memories[1])
        
        # If we're prioritizing weights, keep the weight-sorted order
        # but add a "weight_and_date" field that combines both factors
        if prioritize_weights:
            # Add a combined score that considers both weight and date
            for memory in selected_memories:
                try:
                    # Convert date to a timestamp for numeric comparison
                    date_str = memory.get('date', '1900-01-01')
                    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                    date_score = date_obj.timestamp() / 10000000000  # Normalize to a smaller range
                    
                    # Get weight
                    weight = float(memory.get('weight', 1.0))
                    
                    # Combined score: 70% weight, 30% date recency
                    # This keeps high-weight memories prominent while still respecting chronology somewhat
                    memory['weight_and_date'] = (weight * 0.7) + (date_score * 0.3)
                    
                except Exception as e:
                    # Fallback to just using weight
                    memory['weight_and_date'] = float(memory.get('weight', 1.0))
            
            # Sort by the combined score
            narrative_memories = sorted(
                selected_memories, 
                key=lambda m: m.get('weight_and_date', 0.0),
                reverse=True
            )
        else:
            # Otherwise, sort purely chronologically for traditional timeline narrative
            narrative_memories = sorted(
                selected_memories,
                key=lambda m: m.get('date', '1900-01-01')
            )
            
        # Process images to detect objects if possible
        for memory in narrative_memories:
            if memory.get('filename'):
                # Determine image path based on memory type
                memory_type = memory.get('type', 'user')
                if memory_type == 'user':
                    image_dir = USER_PHOTOS_PATH
                else:
                    image_dir = PUBLIC_PHOTOS_PATH
                    
                image_path = os.path.join(image_dir, memory.get('filename'))
                
                # Detect objects
                if os.path.exists(image_path):
                    _, _, detected_objects = recommendation_engine.detect_objects_in_image(image_path)
                    memory['detected_objects'] = detected_objects
            
        # Generate narrative
        memory_generator = get_synthetic_memory_generator()
        narrative_result = memory_generator.generate_memory_narrative(narrative_memories, query)
        
        # Prepare keywords with types
        keywords = [
            {"text": kw, "type": "primary"} for kw in narrative_result.get('keywords', [])
        ]
        
        # Log information about the narrative generation
        print(f"Generated narrative using {len(narrative_memories)} memories with combined weight of {current_total:.1f}")
        
        return {
            "text": narrative_result.get('text', ''),
            "keywords": keywords,
            "highlighted_terms": narrative_result.get('highlighted_terms', []),
            "source_memories": narrative_result.get('source_memories', [])
        }
        
    except Exception as e:
        print(f"Error generating narrative: {e}")
        return {
            "text": f"An error occurred while generating the narrative: {str(e)}",
            "keywords": [],
            "highlighted_terms": [],
            "source_memories": []
        }

@app.post("/memories/{memory_id}/adjust_weight", response_model=WeightAdjustmentResponse)
def adjust_memory_weight(memory_id: int, adjustment: float = Query(0.1, ge=-1.0, le=1.0)):
    """Adjust the weight of a memory by the specified amount (positive or negative)."""
    try:
        # Connect to the database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Fetch current weight
        cursor.execute("SELECT weight FROM memories WHERE id = ?", (memory_id,))
        result = cursor.fetchone()
        
        if result is None:
            conn.close()
            raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found")
        
        current_weight = result[0] or 1.0
        new_weight = current_weight + adjustment
        
        # Ensure weight stays within reasonable bounds
        new_weight = max(0.1, min(5.0, new_weight))
        
        # Update weight
        cursor.execute(
            "UPDATE memories SET weight = ? WHERE id = ?",
            (new_weight, memory_id)
        )
        
        conn.commit()
        conn.close()
        
        return {
            "status": "success", 
            "message": f"Memory {memory_id} weight adjusted",
            "previous_weight": current_weight,
            "new_weight": new_weight
        }
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# Keep these endpoints for backward compatibility, but implement them using the adjust_memory_weight function
@app.post("/memories/{memory_id}/increase_weight", response_model=WeightAdjustmentResponse)
def increase_weight_memory(memory_id: int):
    """Increase a memory's weight."""
    return adjust_memory_weight(memory_id, 0.1)
    
@app.post("/memories/{memory_id}/decrease_weight", response_model=WeightAdjustmentResponse)
def decrease_weight_memory(memory_id: int):
    """Decrease a memory's weight."""
    return adjust_memory_weight(memory_id, -0.1)



# Test Routes for Image Semantics Detection

@app.get("/test/detect-objects")
def test_detect_objects(
    image_path: str = Query(..., description="Path to the image file to analyze"),
    conf_threshold: float = Query(0.25, description="Confidence threshold for object detection")
):
    """
    Test the YOLO object detection functionality.
    
    This endpoint detects objects in an image using YOLO and returns the results
    with an annotated image showing bounding boxes.
    
    Example usage: /test/detect-objects?image_path=data/processed/user_photos/example.jpg
    """
    try:
        # Get the recommendation engine instance
        engine = recommendation_engine
        
        # Detect objects
        results, annotated_image, detected_objects = engine.detect_objects_in_image(
            image_path, conf_threshold=conf_threshold
        )
        
        if not detected_objects:
            return {
                "status": "warning",
                "message": "No objects detected with sufficient confidence",
                "image_path": image_path,
                "threshold": conf_threshold
            }
        
        # Format and return the results
        return {
            "status": "success",
            "image_path": image_path,
            "detected_objects": detected_objects,
            "annotated_image_base64": annotated_image,
            "object_count": sum(obj.get('count', 1) for obj in detected_objects)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "image_path": image_path
        }

@app.get("/test/image-analysis")
def test_comprehensive_image_analysis(
    image_path: str = Query(..., description="Path to the image file to analyze"),
    output_path: str = Query(None, description="Optional path to save the annotated image")
):
    """
    Test comprehensive image analysis including:
    - Object detection with YOLO
    - Feature extraction with ResNet
    - Visual description generation
    
    Example usage: /test/image-analysis?image_path=data/processed/user_photos/example.jpg
    """
    try:
        # Get the recommendation engine instance
        engine = recommendation_engine
        
        # Extract features and detect objects
        analysis_results = engine.extract_image_with_objects(image_path, output_path)
        
        if not analysis_results["success"]:
            return {
                "status": "warning",
                "message": "Analysis completed but with limited results",
                "image_path": image_path,
                "results": analysis_results
            }
        
        # Generate a descriptive summary of the image content
        description_results = engine.describe_visual_content(image_path)
        
        # Return the combined results
        return {
            "status": "success",
            "image_path": image_path,
            "features": {
                "shape": analysis_results["features"].shape if analysis_results["features"] is not None else None,
                "sample": analysis_results["features"][:5].tolist() if analysis_results["features"] is not None else None
            },
            "detected_objects": analysis_results["objects_detected"],
            "description": description_results["description"],
            "annotated_image_base64": analysis_results["annotated_image_base64"],
            "output_saved": output_path is not None and os.path.exists(output_path) if output_path else False
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "image_path": image_path
        }


@app.get("/test/find-similar-images")
def test_find_similar_images(
    query_image: str = Query(..., description="Path to the query image"),
    dataset_folder: str = Query(..., description="Path to the folder containing images to search"),
    top_n: int = Query(5, description="Number of similar images to return")
):
    """
    Test finding similar images using ResNet50 features.
    
    Example usage: /test/find-similar-images?query_image=data/processed/user_photos/example.jpg&dataset_folder=data/processed/user_photos&top_n=5
    """
    try:
        import os
        
        # Get dataset paths
        if not os.path.exists(dataset_folder):
            return {
                "status": "error",
                "message": f"Dataset folder not found: {dataset_folder}"
            }
            
        # Get all image files in the dataset folder
        dataset_paths = []
        for filename in os.listdir(dataset_folder):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                dataset_paths.append(os.path.join(dataset_folder, filename))
                
        if not dataset_paths:
            return {
                "status": "error",
                "message": f"No image files found in dataset folder: {dataset_folder}"
            }
        
        # Find similar images
        similar_images = recommendation_engine.find_similar_images(
            query_image, dataset_paths, top_n=top_n
        )
        
        # Format results
        formatted_results = []
        for path, similarity in similar_images:
            formatted_results.append({
                "image_path": path,
                "similarity": float(similarity),
                "file_name": os.path.basename(path)
            })
            
        return {
            "status": "success",
            "query_image": query_image,
            "dataset_size": len(dataset_paths),
            "similar_images": formatted_results
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.get("/test/text-similarity")
def test_text_similarity(
    query_text: str = Query(..., description="Text to use as query"),
    memory_type: str = Query('user', description="Type of memories to search: 'user' or 'public'"),
    top_n: int = Query(5, description="Number of similar memories to return")
):
    """
    Test text-based memory similarity using TF-IDF and cosine similarity.
    
    Example usage: /test/text-similarity?query_text=beach sunset&memory_type=user&top_n=5
    """
    try:
        # Create a mock source memory
        source_memory = {
            "id": 0,  # Temporary ID
            "title": "Query Memory",
            "description": query_text,
            "keywords": query_text.split()
        }
        
        # Get all memories of the specified type
        all_memories = get_memories_by_type(memory_type)
        
        if not all_memories:
            return {
                "status": "error",
                "message": f"No memories found of type: {memory_type}"
            }
        
        # Find similar memories
        similar_memories = recommendation_engine.find_similar_memories(
            source_memory, all_memories, top_n=top_n
        )
        
        # Format results
        formatted_results = []
        for memory in similar_memories:
            formatted_results.append({
                "id": memory.get('id'),
                "title": memory.get('title'),
                "location": memory.get('location'),
                "date": memory.get('date'),
                "preview": memory.get('description', '')[:100] + '...' if memory.get('description') else '',
                "keywords": memory.get('keywords', [])
            })
            
        return {
            "status": "success",
            "query_text": query_text,
            "memory_type": memory_type,
            "total_memories_searched": len(all_memories),
            "similar_memories": formatted_results
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.get("/test/memory-stats")
def test_memory_stats():
    """
    Get statistics about the memories in the database.
    
    Example usage: /test/memory-stats
    """
    try:
        # Connect to the database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get counts by type
        cursor.execute("SELECT type, COUNT(*) FROM memories GROUP BY type")
        type_counts = {type_name: count for type_name, count in cursor.fetchall()}
        
        # Get average weights by type
        cursor.execute("SELECT type, AVG(weight) FROM memories GROUP BY type")
        type_avg_weights = {type_name: float(avg_weight) for type_name, avg_weight in cursor.fetchall()}
        
        # Get date ranges
        cursor.execute("SELECT type, MIN(date), MAX(date) FROM memories GROUP BY type")
        date_ranges = {type_name: {"earliest": min_date, "latest": max_date} 
                      for type_name, min_date, max_date in cursor.fetchall()}
        
        # Get top locations
        cursor.execute("""
            SELECT location, COUNT(*) as count
            FROM memories
            GROUP BY location
            ORDER BY count DESC
            LIMIT 5
        """)
        top_locations = [{"location": location, "count": count} for location, count in cursor.fetchall()]
        
        # Close connection
        conn.close()
        
        return {
            "status": "success",
            "counts_by_type": type_counts,
            "average_weights_by_type": type_avg_weights,
            "date_ranges": date_ranges,
            "top_locations": top_locations,
            "database_path": DB_PATH
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)