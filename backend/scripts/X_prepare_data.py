import os
import json
import sys
import argparse
from pathlib import Path
from datetime import datetime
import sqlite3

# Add the parent directory to the path to import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.clip_service import get_clip_service

def extract_image_metadata(image_dir, metadata_file=None):
    """
    Extract metadata from image files in a directory.
    Returns a dict mapping relative image paths to metadata.
    """
    # This is a simplified version - in a real app, you'd use libraries like
    # exifread or PIL to extract EXIF data including GPS coordinates
    
    image_dir = Path(image_dir)
    metadata = {}
    
    # Load existing metadata if provided
    if metadata_file and os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    
    # Find all images
    image_paths = list(image_dir.glob("**/*.jpg")) + list(image_dir.glob("**/*.jpeg")) + list(image_dir.glob("**/*.png"))
    
    for img_path in image_paths:
        rel_path = str(img_path.relative_to(image_dir))
        
        # Skip if we already have metadata for this image
        if rel_path in metadata:
            continue
        
        # Try to extract folder name as location (simplified approach)
        folder_name = img_path.parent.name
        
        # Use file modification time as date (simplified approach)
        mod_time = os.path.getmtime(img_path)
        date_str = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d')
        
        metadata[rel_path] = {
            'location': folder_name,
            'date': date_str
        }
    
    # Save metadata
    if metadata_file:
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    return metadata

def import_images_to_db(db_path, image_dir, metadata_file, memory_type="user"):
    """Import processed images and metadata to the database."""
    # Get CLIP service
    clip_service = get_clip_service()
    
    # Process images and get features
    results = clip_service.batch_process_images(image_dir, metadata_file=metadata_file)
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create table if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS memories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        location TEXT NOT NULL,
        date TEXT NOT NULL,
        type TEXT NOT NULL,
        keywords TEXT,
        content TEXT,
        embedding BLOB,
        image_path TEXT
    )
    ''')
    
    # Insert or update memories
    for result in results:
        # Generate a title from location and keywords
        location = result['location'] or 'Unknown'
        keywords = result['keywords'][:3]
        title = f"{location} - {', '.join(keywords)}" if keywords else location
        
        # Convert features to bytes for storage
        embedding = bytes(json.dumps(result['features']), 'utf-8')
        
        # Generate some content from keywords
        content = f"Image taken at {location}. Features: {', '.join(result['keywords'][:7])}"
        
        # Check if this image is already in the database
        cursor.execute(
            "SELECT id FROM memories WHERE image_path = ? AND type = ?",
            (result['path'], memory_type)
        )
        existing = cursor.fetchone()
        
        if existing:
            # Update existing record
            cursor.execute(
                """
                UPDATE memories 
                SET title = ?, location = ?, date = ?, keywords = ?, content = ?, embedding = ? 
                WHERE id = ?
                """,
                (
                    title, 
                    result['location'], 
                    result['date'], 
                    json.dumps(result['keywords']), 
                    content,
                    embedding,
                    existing[0]
                )
            )
        else:
            # Insert new record
            cursor.execute(
                """
                INSERT INTO memories 
                (title, location, date, type, keywords, content, embedding, image_path) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    title,
                    result['location'],
                    result['date'],
                    memory_type,
                    json.dumps(result['keywords']),
                    content,
                    embedding,
                    result['path']
                )
            )
    
    conn.commit()
    conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process images and prepare database')
    parser.add_argument('--user-dir', type=str, help='Directory containing user images')
    parser.add_argument('--public-dir', type=str, help='Directory containing public images')
    parser.add_argument('--db-path', type=str, default='memories.db', help='Path to SQLite database')
    parser.add_argument('--metadata-file', type=str, help='Path to metadata JSON file')
    
    args = parser.parse_args()
    
    if args.user_dir:
        print(f"Processing user images from {args.user_dir}...")
        metadata = extract_image_metadata(args.user_dir, args.metadata_file)
        import_images_to_db(args.db_path, args.user_dir, args.metadata_file, "user")
    
    if args.public_dir:
        print(f"Processing public images from {args.public_dir}...")
        metadata = extract_image_metadata(args.public_dir, args.metadata_file)
        import_images_to_db(args.db_path, args.public_dir, args.metadata_file, "public")
    
    print("Done!")