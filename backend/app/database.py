# backend/app/database.py
import sqlite3
import json
from datetime import datetime
import os
from pathlib import Path

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
        description TEXT,
        filename TEXT,
        weight REAL DEFAULT 1.0,
        image_path TEXT,
        embedding BLOB
    )
    ''')
    
    conn.commit()
    conn.close()

# Get all memories of a specific type
def get_memories(memory_type="user"):
    """Get all memories of a specific type (user or public)."""
    # Print debug information
    print(f"Getting memories for type: {memory_type}")
    
    conn = sqlite3.connect('memories.db')
    print("yes")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Count total records in the table
    cursor.execute("SELECT COUNT(*) FROM memories")
    total_count = cursor.fetchone()[0]
    print(f"Total records in database: {total_count}")
    
    # Count records of the requested type
    cursor.execute("SELECT COUNT(*) FROM memories WHERE type = ?", (memory_type,))
    type_count = cursor.fetchone()[0]
    print(f"Records of type '{memory_type}': {type_count}")
    
    # Get column names for debugging
    cursor.execute("PRAGMA table_info(memories)")
    column_info = cursor.fetchall()
    column_names = [col[1] for col in column_info]
    print(f"Table columns: {column_names}")
    
    # Execute the query
    try:
        cursor.execute(
            """
            SELECT id, title, location, date, type, keywords, 
                   description, filename, weight, image_path
            FROM memories WHERE type = ?
            LIMIT 5
            """, 
            (memory_type,)
        )
        results = cursor.fetchall()
        print(f"Query returned {len(results)} results")
        
        # Debug: Print the first result if available
        if results:
            first_row = dict(results[0])
            print(f"First row: {first_row}")
        
        memories = []
        for row in results:
            memory = {
                "id": row['id'],
                "title": row['title'],
                "location": row['location'],
                "date": row['date'],
                "type": row['type'],
                "keywords": json.loads(row['keywords']) if row['keywords'] else [],
                "description": row['description'],
                "filename": row['filename'],
                "weight": row['weight'],
                "image_path": row['image_path']
            }
            memories.append(memory)
        
        conn.close()
        return memories
        
    except Exception as e:
        print(f"Error in get_memories: {e}")
        conn.close()
        return []

def get_db_info():
    """Get information about the database."""
    try:
        db_path = os.path.abspath('memories.db')
        exists = os.path.exists(db_path)
        
        info = {
            "path": db_path,
            "exists": exists,
            "size": os.path.getsize(db_path) if exists else 0
        }
        
        if exists:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM memories")
            info["total_records"] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM memories WHERE type = 'user'")
            info["user_records"] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM memories WHERE type = 'public'")
            info["public_records"] = cursor.fetchone()[0]
            
            conn.close()
        
        return info
    except Exception as e:
        return {"error": str(e)}
    
    
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
import json
import os
import sqlite3
from pathlib import Path
def seed_data(data_dir="D:/ahYen Workspace/ahYen Work/CMU_academic/MSCD_Y1_2425/48632-Taxonavigating the Digital Space/memory-map/data"):
    """Seed the database with data from metadata files."""
    # Create base data directory path
    data_path = Path(data_dir)
    
    # Define paths to metadata files
    metadata_dir = data_path / "metadata"
    public_metadata_file = metadata_dir / "public_metadata.json"
    user_metadata_file = metadata_dir / "user_metadata.json"
    
    # Check if metadata files exist
    if not public_metadata_file.exists():
        print(f"Error: Public metadata file not found at {public_metadata_file}")
        return False
    if not user_metadata_file.exists():
        print(f"Error: User metadata file not found at {user_metadata_file}")
        return False
        
    print(f"Found metadata files at {metadata_dir}")
    
    # Connect to database
    db_path = 'memories.db'  # You might need to adjust this path
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # First check if we've already seeded data
    cursor.execute("SELECT COUNT(*) FROM memories")
    count = cursor.fetchone()[0]
    
    if count > 0:
        conn.close()
        print(f"Database at {db_path} already contains {count} records. Skipping seed operation.")
        return False  # Data already exists
    
    # Check table structure
    cursor.execute("PRAGMA table_info(memories)")
    columns = [column[1] for column in cursor.fetchall()]
    print(f"Database columns: {columns}")
    
    # Load metadata from files
    try:
        with open(public_metadata_file, 'r', encoding='utf-8') as f:
            public_metadata = json.load(f)
        
        with open(user_metadata_file, 'r', encoding='utf-8') as f:
            user_metadata = json.load(f)
        
        public_count = len(public_metadata)
        user_count = len(user_metadata)
        total_count = public_count + user_count
        
        print(f"Found {public_count} public memories and {user_count} user memories.")
        print(f"Total: {total_count} memories to import.")
        
    except Exception as e:
        print(f"Error loading metadata files: {e}")
        conn.close()
        return False
    
    # Process public memories
    public_inserted = 0
    for filename, metadata in public_metadata.items():
        try:
            # Extract data from metadata
            location = metadata.get('location', '')
            date = metadata.get('date', '')
            keywords = metadata.get('keywords', [])
            weight = metadata.get('weight', 1.0)
            description = metadata.get('description', '')
            
            # Generate a title from the filename
            title = filename.replace('.jpg', '').replace('_', ' ').title()
            
            # Construct image path
            image_path = f"/images/public/{filename}"
            
            # Adjust column names based on what exists in the database
            column_names = []
            values = []
            
            # Always include these required columns
            column_names.extend(['title', 'location', 'date', 'type'])
            values.extend([title, location, date, 'public'])
            
            # Add optional columns if they exist in the database
            if 'keywords' in columns:
                column_names.append('keywords')
                values.append(json.dumps(keywords))
                
            if 'description' in columns:
                column_names.append('description')
                values.append(description)
                
            if 'filename' in columns:
                column_names.append('filename')
                values.append(filename)
                
            if 'weight' in columns:
                column_names.append('weight')
                values.append(weight)
                
            if 'image_path' in columns:
                column_names.append('image_path')
                values.append(image_path)
                
            if 'embedding' in columns and 'embedding' in metadata:
                column_names.append('embedding')
                values.append(json.dumps(metadata['embedding']))
            
            # Build the SQL insert statement dynamically
            placeholders = ','.join(['?'] * len(column_names))
            sql = f"INSERT INTO memories ({','.join(column_names)}) VALUES ({placeholders})"
            
            # Execute the insert
            cursor.execute(sql, values)
            public_inserted += 1
            
            if public_inserted <= 3 or public_inserted % 20 == 0:
                print(f"Inserted public memory {public_inserted}/{public_count}: {filename}")
                
        except Exception as e:
            print(f"Error inserting public memory {filename}: {e}")
    
    # Process user memories (similar code as above)
    user_inserted = 0
    for filename, metadata in user_metadata.items():
        try:
            # Extract data from metadata
            location = metadata.get('location', '')
            date = metadata.get('date', '')
            keywords = metadata.get('keywords', [])
            weight = metadata.get('weight', 1.0)
            description = metadata.get('description', '')
            
            # Generate a title from the filename
            title = filename.replace('.jpg', '').replace('_', ' ').title()
            
            # Construct image path
            image_path = f"/images/user/{filename}"
            
            # Adjust column names based on what exists in the database
            column_names = []
            values = []
            
            # Always include these required columns
            column_names.extend(['title', 'location', 'date', 'type'])
            values.extend([title, location, date, 'user'])
            
            # Add optional columns if they exist in the database
            if 'keywords' in columns:
                column_names.append('keywords')
                values.append(json.dumps(keywords))
                
            if 'description' in columns:
                column_names.append('description')
                values.append(description)
                
            if 'filename' in columns:
                column_names.append('filename')
                values.append(filename)
                
            if 'weight' in columns:
                column_names.append('weight')
                values.append(weight)
                
            if 'image_path' in columns:
                column_names.append('image_path')
                values.append(image_path)
                
            if 'embedding' in columns and 'embedding' in metadata:
                column_names.append('embedding')
                values.append(json.dumps(metadata['embedding']))
            
            # Build the SQL insert statement dynamically
            placeholders = ','.join(['?'] * len(column_names))
            sql = f"INSERT INTO memories ({','.join(column_names)}) VALUES ({placeholders})"
            
            # Execute the insert
            cursor.execute(sql, values)
            user_inserted += 1
            
            if user_inserted <= 3 or user_inserted % 20 == 0:
                print(f"Inserted user memory {user_inserted}/{user_count}: {filename}")
                
        except Exception as e:
            print(f"Error inserting user memory {filename}: {e}")
    
    conn.commit()
    conn.close()
    
    total_inserted = public_inserted + user_inserted
    print(f"Successfully seeded database with {total_inserted} memories:")
    print(f"- {public_inserted} public memories")
    print(f"- {user_inserted} user memories")
    
    return True