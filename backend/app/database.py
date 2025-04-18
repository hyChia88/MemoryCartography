import sqlite3
import json
from datetime import datetime

# Database initialization function
def init_db():
    conn = sqlite3.connect('memories.db')
    cursor = conn.cursor()
    
    # Create memories table if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS memories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        location TEXT NOT NULL,
        date TEXT NOT NULL,
        type TEXT NOT NULL,
        keywords TEXT,
        content TEXT,
        embedding BLOB
    )
    ''')
    
    conn.commit()
    conn.close()

# Get all memories of a specific type
def get_memories(memory_type="user"):
    conn = sqlite3.connect('memories.db')
    conn.row_factory = sqlite3.Row  # This enables column access by name
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT id, title, location, date, keywords, content FROM memories WHERE type = ?", 
        (memory_type,)
    )
    results = cursor.fetchall()
    conn.close()
    
    memories = []
    for row in results:
        keywords = json.loads(row['keywords']) if row['keywords'] else []
        memories.append({
            "id": row['id'],
            "title": row['title'],
            "location": row['location'],
            "date": row['date'],
            "keywords": keywords,
            "content": row['content']
        })
    
    return memories

# Add a new memory
def add_memory(title, location, date, memory_type, keywords=None, content=None, embedding=None):
    conn = sqlite3.connect('memories.db')
    cursor = conn.cursor()
    
    # Serialize keywords to JSON if provided
    keywords_json = json.dumps(keywords) if keywords else None
    
    cursor.execute(
        "INSERT INTO memories (title, location, date, type, keywords, content, embedding) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (title, location, date, memory_type, keywords_json, content, embedding)
    )
    
    conn.commit()
    memory_id = cursor.lastrowid
    conn.close()
    
    return memory_id

# Find memories by location
def find_memories_by_location(location, memory_type="user"):
    conn = sqlite3.connect('memories.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Search for memories with matching location
    cursor.execute(
        "SELECT id, title, location, date, keywords, content FROM memories WHERE type = ? AND location LIKE ?", 
        (memory_type, f"%{location}%")
    )
    results = cursor.fetchall()
    
    # If no results, search in keywords
    if not results:
        cursor.execute("SELECT id, title, location, date, keywords, content FROM memories WHERE type = ?", (memory_type,))
        all_results = cursor.fetchall()
        
        results = []
        for row in all_results:
            if not row['keywords']:
                continue
                
            keywords = json.loads(row['keywords'])
            if any(location.lower() in kw.lower() for kw in keywords):
                results.append(row)
    
    conn.close()
    
    memories = []
    for row in results:
        keywords = json.loads(row['keywords']) if row['keywords'] else []
        memories.append({
            "id": row['id'],
            "title": row['title'],
            "location": row['location'],
            "date": row['date'],
            "keywords": keywords,
            "content": row['content']
        })
    
    return memories

# Seed the database with sample data
def seed_data():
    conn = sqlite3.connect('memories.db')
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
            "title": "Summer in New York",
            "location": "New York",
            "date": "2022-07-15",
            "type": "user",
            "keywords": ["skyline", "central park", "hot dog", "taxi", "sunset"],
            "content": "I remember walking through Central Park on a hot summer day. The skyline was beautiful against the setting sun."
        },
        {
            "title": "Sunset at Golden Gate",
            "location": "San Francisco",
            "date": "2021-09-22",
            "type": "user",
            "keywords": ["bridge", "foggy", "ocean", "sunset", "cold"],
            "content": "The Golden Gate Bridge looked stunning with the fog rolling in. The ocean breeze was cold but refreshing."
        },
        {
            "title": "Winter Walk",
            "location": "Chicago",
            "date": "2023-01-10",
            "type": "user",
            "keywords": ["snow", "wind", "lake", "cold", "architecture"],
            "content": "The wind coming off Lake Michigan was freezing, but the snow-covered architecture was worth it."
        }
    ]
    
    # Sample public memories
    public_memories = [
        {
            "title": "City Lights",
            "location": "Tokyo",
            "date": "2022-11-05",
            "type": "public",
            "keywords": ["neon", "crowds", "ramen", "skyscraper", "train"],
            "content": "The neon lights of Tokyo create a mesmerizing landscape at night. The city never seems to sleep."
        },
        {
            "title": "Beach Memories",
            "location": "Bali",
            "date": "2022-05-18",
            "type": "public",
            "keywords": ["waves", "sand", "palm trees", "sunset", "warm"],
            "content": "The beaches in Bali offer the perfect combination of warm sun, gentle waves, and swaying palm trees."
        },
        {
            "title": "Mountain View",
            "location": "Swiss Alps",
            "date": "2023-02-03",
            "type": "public",
            "keywords": ["snow", "peaks", "hiking", "fresh air", "view"],
            "content": "The breathtaking view of snow-capped peaks in the Swiss Alps makes you feel small yet connected to something greater."
        }
    ]
    
    # Insert user memories
    for memory in user_memories:
        cursor.execute(
            "INSERT INTO memories (title, location, date, type, keywords, content) VALUES (?, ?, ?, ?, ?, ?)",
            (
                memory["title"], 
                memory["location"], 
                memory["date"], 
                memory["type"], 
                json.dumps(memory["keywords"]),
                memory["content"]
            )
        )
    
    # Insert public memories
    for memory in public_memories:
        cursor.execute(
            "INSERT INTO memories (title, location, date, type, keywords, content) VALUES (?, ?, ?, ?, ?, ?)",
            (
                memory["title"], 
                memory["location"], 
                memory["date"], 
                memory["type"], 
                json.dumps(memory["keywords"]),
                memory["content"]
            )
        )
    
    conn.commit()
    conn.close()
    return True  # Successfully seeded