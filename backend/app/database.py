# backend/app/database.py
import sqlite3
import json
from datetime import datetime
import os
from pathlib import Path

# Database initialization function
def init_db(db_path='data/metadata/memories.db'):
    """Initialize the database with necessary tables."""
    # Ensure the directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create memories table if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS memories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        title TEXT NOT NULL,
        location TEXT NOT NULL,
        date TEXT NOT NULL,
        type TEXT NOT NULL,
        keywords TEXT,
        description TEXT,
        weight REAL DEFAULT 1.0,
        embedding TEXT
    )
    ''')
    
    # Create user_interactions table to track user clicks and interactions
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_interactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        memory_id INTEGER NOT NULL,
        interaction_type TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        FOREIGN KEY (memory_id) REFERENCES memories (id)
    )
    ''')
    
    conn.commit()
    conn.close()

# Get all memories of a specific type
def get_memories(memory_type="user", db_path='data/metadata/memories.db'):
    """Get all memories of a specific type."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # This enables column access by name
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT id, filename, title, location, date, keywords, description, weight FROM memories WHERE type = ?", 
        (memory_type,)
    )
    results = cursor.fetchall()
    conn.close()
    
    memories = []
    for row in results:
        keywords = json.loads(row['keywords']) if row['keywords'] else []
        memories.append({
            "id": row['id'],
            "filename": row['filename'],
            "title": row['title'],
            "location": row['location'],
            "date": row['date'],
            "keywords": keywords,
            "description": row['description'],
            "weight": float(row['weight'] or 1.0)
        })
    
    return memories

# Add a new memory
def add_memory(filename, title, location, date, memory_type, 
               keywords=None, description=None, weight=1.0, embedding=None, 
               db_path='data/metadata/memories.db'):
    """Add a new memory to the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Serialize keywords to JSON if provided
    keywords_json = json.dumps(keywords) if keywords else None
    embedding_json = json.dumps(embedding) if embedding else None
    
    cursor.execute(
        '''
        INSERT INTO memories 
        (filename, title, location, date, type, keywords, description, weight, embedding) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''',
        (filename, title, location, date, memory_type, keywords_json, description, weight, embedding_json)
    )
    
    conn.commit()
    memory_id = cursor.lastrowid
    conn.close()
    
    return memory_id

# Update memory weight
def update_memory_weight(memory_id, new_weight=None, increment=0.1, db_path='data/metadata/memories.db'):
    """Update the weight of a memory."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    if new_weight is None:
        # Get current weight and increment it
        cursor.execute("SELECT weight FROM memories WHERE id = ?", (memory_id,))
        result = cursor.fetchone()
        
        if not result:
            conn.close()
            return False
            
        current_weight = result[0] or 1.0
        new_weight = current_weight + increment
    
    # Update the weight
    cursor.execute(
        "UPDATE memories SET weight = ? WHERE id = ?",
        (new_weight, memory_id)
    )
    
    # Record the interaction
    timestamp = datetime.now().isoformat()
    cursor.execute(
        "INSERT INTO user_interactions (memory_id, interaction_type, timestamp) VALUES (?, ?, ?)",
        (memory_id, "weight_update", timestamp)
    )
    
    conn.commit()
    conn.close()
    return True

# Record a memory interaction (view, click, etc.)
def record_memory_interaction(memory_id, interaction_type, db_path='data/metadata/memories.db'):
    """Record a user interaction with a memory."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    timestamp = datetime.now().isoformat()
    cursor.execute(
        "INSERT INTO user_interactions (memory_id, interaction_type, timestamp) VALUES (?, ?, ?)",
        (memory_id, interaction_type, timestamp)
    )
    
    # If the interaction is a click/view, slightly increase the weight
    if interaction_type in ['click', 'view', 'select']:
        cursor.execute("SELECT weight FROM memories WHERE id = ?", (memory_id,))
        result = cursor.fetchone()
        
        if result:
            current_weight = result[0] or 1.0
            # Smaller increment for normal interactions
            new_weight = current_weight + 0.05
            cursor.execute(
                "UPDATE memories SET weight = ? WHERE id = ?",
                (new_weight, memory_id)
            )
    
    conn.commit()
    conn.close()
    return True

# Get a specific memory by ID
def get_memory_by_id(memory_id, db_path='data/metadata/memories.db'):
    """Get a specific memory by its ID."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute(
        """
        SELECT id, filename, title, location, date, type, keywords, description, weight, embedding
        FROM memories 
        WHERE id = ?
        """,
        (memory_id,)
    )
    result = cursor.fetchone()
    conn.close()
    
    if not result:
        return None
    
    memory = dict(result)
    memory['keywords'] = json.loads(memory['keywords']) if memory['keywords'] else []
    memory['embedding'] = json.loads(memory['embedding']) if memory['embedding'] else None
    memory['weight'] = float(memory['weight'] or 1.0)
    
    # Add image path based on type
    base_dir = Path("data/processed")
    if memory['type'] == 'user':
        memory['image_path'] = str(base_dir / "user_photos" / memory['filename'])
    else:
        memory['image_path'] = str(base_dir / "public_photos" / memory['filename'])
    
    return memory

# Find memories by location
def find_memories_by_location(location, memory_type="user", limit=10, db_path='data/metadata/memories.db'):
    """Find memories by location."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Search for memories with matching location
    cursor.execute(
        """
        SELECT id, filename, title, location, date, type, keywords, description, weight
        FROM memories 
        WHERE type = ? AND location LIKE ?
        ORDER BY weight DESC
        LIMIT ?
        """, 
        (memory_type, f"%{location}%", limit)
    )
    results = cursor.fetchall()
    
    # If no results, search in keywords
    if not results:
        cursor.execute(
            """
            SELECT id, filename, title, location, date, type, keywords, description, weight
            FROM memories 
            WHERE type = ?
            ORDER BY weight DESC
            """, 
            (memory_type,)
        )
        all_results = cursor.fetchall()
        
        matching_results = []
        for row in all_results:
            if not row['keywords']:
                continue
                
            keywords = json.loads(row['keywords'])
            if any(location.lower() in kw.lower() for kw in keywords):
                matching_results.append(row)
                
        results = matching_results[:limit]
    
    conn.close()
    
    # Process and return memories
    memories = []
    base_dir = Path("data/processed")
    
    for row in results:
        memory = dict(row)
        memory['keywords'] = json.loads(memory['keywords']) if memory['keywords'] else []
        memory['weight'] = float(memory['weight'] or 1.0)
        
        # Add image path
        if memory['type'] == 'user':
            memory['image_path'] = str(base_dir / "user_photos" / memory['filename'])
        else:
            memory['image_path'] = str(base_dir / "public_photos" / memory['filename'])
        
        memories.append(memory)
    
    return memories

# Seed the database with sample data (for testing)
def seed_data(db_path='data/metadata/memories.db'):
    """Seed the database with sample data."""
    # Initialize the database
    init_db(db_path)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # First check if we've already seeded data
    cursor.execute("SELECT COUNT(*) FROM memories")
    count = cursor.fetchone()[0]
    
    if count > 0:
        conn.close()
        return False  # Data already exists
    
    # Sample user memories
    user_memories = [
        {
            "filename": "user_0001.jpg",
            "title": "Summer in New York",
            "location": "New York City, USA",
            "date": "2022-07-15",
            "type": "user",
            "keywords": ["skyline", "central park", "hot dog", "taxi", "sunset", "urban", "skyscraper", "busy", "exciting"],
            "description": "I remember walking through Central Park on a hot summer day. The skyline was beautiful against the setting sun.",
            "weight": 1.2
        },
        {
            "filename": "user_0002.jpg",
            "title": "Sunset at Golden Gate",
            "location": "San Francisco, USA",
            "date": "2021-09-22",
            "type": "user",
            "keywords": ["bridge", "foggy", "ocean", "sunset", "cold", "windy", "orange", "engineering", "iconic"],
            "description": "The Golden Gate Bridge looked stunning with the fog rolling in. The ocean breeze was cold but refreshing.",
            "weight": 1.0
        },
        {
            "filename": "user_0003.jpg",
            "title": "Winter Walk",
            "location": "Chicago, USA",
            "date": "2023-01-10",
            "type": "user",
            "keywords": ["snow", "wind", "lake", "cold", "architecture", "frozen", "white", "peaceful", "crisp"],
            "description": "The wind coming off Lake Michigan was freezing, but the snow-covered architecture was worth it.",
            "weight": 0.8
        },
        {
            "filename": "user_0004.jpg",
            "title": "Breakfast in Bentong",
            "location": "Bentong, Malaysia",
            "date": "2023-05-10",
            "type": "user",
            "keywords": ["food", "morning", "noodles", "coffee", "traditional", "local", "delicious", "steamy", "aromatic"],
            "description": "Started the day with a delicious local breakfast. The noodles were perfectly cooked and the coffee was strong.",
            "weight": 1.1
        },
        {
            "filename": "user_0005.jpg",
            "title": "Night Market Adventure",
            "location": "Kuala Lumpur, Malaysia",
            "date": "2023-05-12",
            "type": "user",
            "keywords": ["market", "night", "food", "shopping", "colorful", "busy", "vibrant", "street food", "lively"],
            "description": "The night market was bustling with activity. So many different foods to try and items to see.",
            "weight": 1.3
        }
    ]
    
    # Sample public memories
    public_memories = [
        {
            "filename": "public_0001.jpg",
            "title": "City Lights",
            "location": "Tokyo, Japan",
            "date": "2022-11-05",
            "type": "public",
            "keywords": ["neon", "crowds", "ramen", "skyscraper", "train", "busy", "nightlife", "technology", "vibrant"],
            "description": "The neon lights of Tokyo create a mesmerizing landscape at night. The city never seems to sleep.",
            "weight": 1.0
        },
        {
            "filename": "public_0002.jpg",
            "title": "Beach Memories",
            "location": "Bali, Indonesia",
            "date": "2022-05-18",
            "type": "public",
            "keywords": ["waves", "sand", "palm trees", "sunset", "warm", "tropical", "relaxing", "ocean", "paradise"],
            "description": "The beaches in Bali offer the perfect combination of warm sun, gentle waves, and swaying palm trees.",
            "weight": 1.1
        },
        {
            "filename": "public_0003.jpg",
            "title": "Mountain View",
            "location": "Swiss Alps, Switzerland",
            "date": "2023-02-03",
            "type": "public",
            "keywords": ["snow", "peaks", "hiking", "fresh air", "view", "majestic", "panoramic", "pristine", "nature"],
            "description": "The breathtaking view of snow-capped peaks in the Swiss Alps makes you feel small yet connected to something greater.",
            "weight": 0.9
        },
        {
            "filename": "public_0004.jpg",
            "title": "Petronas Towers",
            "location": "Kuala Lumpur, Malaysia",
            "date": "2023-04-15",
            "type": "public",
            "keywords": ["towers", "architecture", "modern", "city", "skyscraper", "landmark", "night", "lights", "reflection"],
            "description": "The iconic Petronas Towers dominate the Kuala Lumpur skyline, especially beautiful when lit up at night.",
            "weight": 1.2
        },
        {
            "filename": "public_0005.jpg",
            "title": "Bentong Waterfall",
            "location": "Bentong, Malaysia",
            "date": "2023-03-10",
            "type": "public",
            "keywords": ["waterfall", "nature", "forest", "refreshing", "green", "serene", "peaceful", "water", "outdoor"],
            "description": "The hidden waterfall near Bentong offers a peaceful retreat from the bustling city. The water is crystal clear and refreshing.",
            "weight": 1.0
        }
    ]
    
    # Insert user memories
    for memory in user_memories:
        cursor.execute(
            '''
            INSERT INTO memories 
            (filename, title, location, date, type, keywords, description, weight) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            (
                memory["filename"], 
                memory["title"], 
                memory["location"], 
                memory["date"], 
                memory["type"], 
                json.dumps(memory["keywords"]),
                memory["description"],
                memory["weight"]
            )
        )
    
    # Insert public memories
    for memory in public_memories:
        cursor.execute(
            '''
            INSERT INTO memories 
            (filename, title, location, date, type, keywords, description, weight) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            (
                memory["filename"], 
                memory["title"], 
                memory["location"], 
                memory["date"], 
                memory["type"], 
                json.dumps(memory["keywords"]),
                memory["description"],
                memory["weight"]
            )
        )
    
    conn.commit()
    conn.close()
    return True  # Successfully seeded