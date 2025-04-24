# main.py
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import sqlite3
import os
import json
import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from app.database import search_memories, init_db
from app.services.memory_recommendation import MemoryRecommendationEngine
from fastapi.middleware.cors import CORSMiddleware
from app.services.synthetic_memory_generator import get_synthetic_memory_generator
from fastapi.staticfiles import StaticFiles

# Define database and data paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = 'data/metadata/memories.db'
USER_PHOTOS_PATH = os.path.join('data/processed/user_photos/')
PUBLIC_PHOTOS_PATH = os.path.join('data/processed/public_photos/')

app = FastAPI()
app.mount("/user-photos", StaticFiles(directory=USER_PHOTOS_PATH), name="user-photos")
app.mount("/public-photos", StaticFiles(directory=PUBLIC_PHOTOS_PATH), name="public-photos")

recommendation_engine = MemoryRecommendationEngine()

# Initialize database if it doesn't exist
if not os.path.exists(DB_PATH):
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
                    memory['embedding_vector'] = json.loads(memory['embedding'])
                except json.JSONDecodeError:
                    memory['embedding_vector'] = []
            else:
                memory['embedding_vector'] = []
                
        return memories
    except Exception as e:
        print(f"Error fetching memories of type {memory_type}: {e}")
        return []
    finally:
        if conn:
            conn.close()
        
        

@app.get("/")
def read_root():
    return {"message": "Welcome to the Memory Map API"}

@app.get("/memories/search", response_model=List[Memory])
def search_memories_endpoint(
    query: str,
    memory_type: str = 'user',
    limit: int = 10,
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
        
        # Use the enhanced search function that leverages embeddings
        memories = search_memories_with_embeddings(query, memory_type, limit * 2, db_path=DB_PATH)
        
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
        
        # Sort the memories based on the sort_by parameter
        if sort_by == "date":
            memories = sorted(
                memories, 
                key=lambda m: datetime.strptime(m.get('date', '1900-01-01'), '%Y-%m-%d'),
                reverse=True
            )
        elif sort_by == "relevance":
            # 'relevance' is what the search function already returns
            pass  # keep the order from the search function
        else:  # Default to weight
            memories = sorted(
                memories, 
                key=lambda m: float(m.get('weight', 1.0)),
                reverse=True
            )
        
        # Return only the top 'limit' memories after sorting
        return memories[:limit]
    
    except Exception as e:
        print(f"Error in search endpoint: {e}")
        # Fallback to original search method
        from app.database import search_memories
        
        memories = search_memories(query, memory_type, limit * 2, db_path=DB_PATH)
        
        # Apply the same sorting
        if sort_by == "date":
            memories = sorted(
                memories, 
                key=lambda m: datetime.strptime(m.get('date', '1900-01-01'), '%Y-%m-%d'),
                reverse=True
            )
        else:  # Default to weight
            memories = sorted(
                memories, 
                key=lambda m: float(m.get('weight', 1.0)),
                reverse=True
            )
            
        return memories[:limit]

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

@app.get("/memories/narrative", response_model=NarrativeResponse)
def generate_narrative_endpoint(
    query: str, 
    memory_type: str = 'user',
    use_embedding: bool = True,
    max_memories: int = 10,
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

@app.get("/memories/clusters")
def get_memory_clusters(
    memory_type: str = 'user',
    num_clusters: int = 3
):
    """Group memories into clusters based on embedding similarity."""
    try:
        from sklearn.cluster import KMeans
        
        # Get memories with embeddings
        memories = get_memories_by_type(memory_type)
        
        # Filter memories with valid embeddings
        valid_memories = []
        embeddings = []
        
        for memory in memories:
            if 'embedding_vector' in memory and memory['embedding_vector'] and len(memory['embedding_vector']) > 0:
                embeddings.append(memory['embedding_vector'])
                valid_memories.append(memory)
        
        if len(valid_memories) < num_clusters:
            return {
                "clusters": [valid_memories],
                "cluster_keywords": [["all", "memories"]]
            }
        
        # Apply clustering
        kmeans = KMeans(n_clusters=min(num_clusters, len(valid_memories)))
        embeddings = np.array(embeddings)
        clusters = kmeans.fit_predict(embeddings)
        
        # Group memories by cluster
        memory_clusters = [[] for _ in range(max(clusters) + 1)]
        for i, cluster_id in enumerate(clusters):
            memory_clusters[cluster_id].append(valid_memories[i])
        
        # Generate keywords for each cluster
        cluster_keywords = []
        for cluster in memory_clusters:
            # Collect all keywords from memories in this cluster
            all_keywords = []
            for memory in cluster:
                all_keywords.extend(memory.get('keywords', []))
            
            # Get most frequent keywords
            keyword_counts = {}
            for kw in all_keywords:
                keyword_counts[kw] = keyword_counts.get(kw, 0) + 1
            
            # Sort by frequency
            sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
            top_keywords = [kw for kw, count in sorted_keywords[:5]]
            
            cluster_keywords.append(top_keywords)
        
        return {
            "clusters": memory_clusters,
            "cluster_keywords": cluster_keywords
        }
        
    except Exception as e:
        print(f"Error clustering memories: {e}")
        # Fall back to returning all memories as a single cluster
        memories = get_memories_by_type(memory_type)
        return {
            "clusters": [memories],
            "cluster_keywords": [["all", "memories"]]
        }
        
@app.post("/memories/reset_weights_from_json", response_model=dict)
async def reset_weights_from_json(
    memory_type: str = Query("all", description="Type of memories to reset: 'user', 'public', or 'all'")
):
    """
    Reset memory weights to their original values from JSON metadata files.
    
    Args:
        memory_type: Type of memories to reset weights for ('user', 'public', or 'all')
        
    Returns:
        A dictionary with the count of updated memories and status message
    """
    try:
        # Define paths to the metadata files
        user_metadata_path = os.path.join(os.path.dirname(DB_PATH), "user_metadata.json")
        public_metadata_path = os.path.join(os.path.dirname(DB_PATH), "public_metadata.json")
        
        # Initialize counters and memory maps
        updated_count = 0
        memory_id_to_weight = {}
        processed_files = []
        
        # Load user metadata if needed
        if memory_type.lower() in ["user", "all"] and os.path.exists(user_metadata_path):
            try:
                with open(user_metadata_path, 'r', encoding='utf-8') as f:
                    user_metadata = json.load(f)
                    
                    # Extract filename and weights from the specific format
                    for filename, memory_data in user_metadata.items():
                        # Extract just the base filename without extension to match DB records
                        base_filename = os.path.splitext(filename)[0]
                        weight = memory_data.get('weight', 1.0)
                        
                        # Store the mapping for database update
                        memory_id_to_weight[base_filename] = weight
                    
                processed_files.append(os.path.basename(user_metadata_path))
                print(f"Loaded {len(memory_id_to_weight)} weights from user metadata")
            except json.JSONDecodeError as e:
                print(f"Error parsing user metadata JSON: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error parsing user metadata JSON: {str(e)}"
                )
        
        # Load public metadata if needed
        if memory_type.lower() in ["public", "all"] and os.path.exists(public_metadata_path):
            try:
                with open(public_metadata_path, 'r', encoding='utf-8') as f:
                    public_metadata = json.load(f)
                    
                    # Extract filename and weights from the specific format
                    initial_count = len(memory_id_to_weight)
                    for filename, memory_data in public_metadata.items():
                        # Extract just the base filename without extension to match DB records
                        base_filename = os.path.splitext(filename)[0]
                        weight = memory_data.get('weight', 1.0)
                        
                        # Store the mapping for database update
                        memory_id_to_weight[base_filename] = weight
                    
                processed_files.append(os.path.basename(public_metadata_path))
                print(f"Loaded {len(memory_id_to_weight) - initial_count} weights from public metadata")
            except json.JSONDecodeError as e:
                print(f"Error parsing public metadata JSON: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error parsing public metadata JSON: {str(e)}"
                )
        
        # If no weights were found, return early
        if not memory_id_to_weight:
            return {
                "status": "warning",
                "message": f"No weights found in metadata files for type '{memory_type}'",
                "updated_count": 0,
                "processed_files": processed_files
            }
        
        # Connect to the database to update weights
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # First, get a map of filenames to IDs from the database
        if memory_type.lower() == "all":
            cursor.execute("SELECT id, filename FROM memories")
        else:
            cursor.execute("SELECT id, filename FROM memories WHERE type = ?", (memory_type.lower(),))
        
        filename_to_id = {}
        for row in cursor.fetchall():
            memory_id, filename = row
            if filename:
                # Extract base filename without extension to match our metadata keys
                base_filename = os.path.splitext(os.path.basename(filename))[0]
                filename_to_id[base_filename] = memory_id
        
        # Now update weights based on filename matches
        updated_weights = []
        for base_filename, weight in memory_id_to_weight.items():
            if base_filename in filename_to_id:
                memory_id = filename_to_id[base_filename]
                cursor.execute(
                    "UPDATE memories SET weight = ? WHERE id = ?",
                    (weight, memory_id)
                )
                if cursor.rowcount > 0:
                    updated_weights.append((memory_id, weight))
        
        # Commit changes and get updated count
        conn.commit()
        updated_count = len(updated_weights)
        
        # Print detailed information for debugging
        for memory_id, weight in updated_weights[:5]:  # Print first 5 for brevity
            print(f"Updated memory ID {memory_id} with weight {weight}")
        
        if updated_count > 5:
            print(f"... and {updated_count - 5} more memories")
        
        # Close the connection
        conn.close()
        
        # Return success response
        return {
            "status": "success" if updated_count > 0 else "warning",
            "message": f"Successfully reset weights for {updated_count} memories from metadata files" if updated_count > 0 else "No matching database records found for metadata entries",
            "type": memory_type,
            "updated_count": updated_count,
            "processed_files": processed_files
        }
        
    except sqlite3.Error as e:
        # Log database errors
        print(f"Database error while resetting weights: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Database error: {str(e)}"
        )
        
    except Exception as e:
        # Log unexpected errors
        print(f"Unexpected error while resetting weights: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"An unexpected error occurred: {str(e)}"
        )