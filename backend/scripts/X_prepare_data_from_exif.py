# backend/scripts/prepare_data_from_exif.py
import os
import json
import sys
import argparse
from pathlib import Path
from datetime import datetime
import sqlite3
import exifread
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import requests
from geopy.geocoders import Nominatim

# Add the parent directory to the path to import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import CLIP service (if you have this from earlier code)
try:
    from app.services.clip_service import get_clip_service
except ImportError:
    print("CLIP service not available. Will skip feature extraction.")
    get_clip_service = None

def get_exif_data(image_path):
    """Extract EXIF data from an image file."""
    try:
        # Open the image file for reading (binary mode)
        with open(image_path, 'rb') as f:
            # Return Exif tags
            return exifread.process_file(f, details=False)
    except Exception as e:
        print(f"Error reading EXIF data from {image_path}: {e}")
        return {}

def get_gps_data(exif_data):
    """Extract GPS coordinates from EXIF data."""
    if not exif_data:
        return None
    
    # Check for GPS tags
    gps_latitude = exif_data.get('GPS GPSLatitude')
    gps_latitude_ref = exif_data.get('GPS GPSLatitudeRef')
    gps_longitude = exif_data.get('GPS GPSLongitude')
    gps_longitude_ref = exif_data.get('GPS GPSLongitudeRef')
    
    if not (gps_latitude and gps_latitude_ref and gps_longitude and gps_longitude_ref):
        return None
    
    # Convert the GPS coordinates to decimal degrees
    try:
        lat_value = _convert_to_degrees(gps_latitude.values)
        if gps_latitude_ref.values == 'S':
            lat_value = -lat_value
            
        lon_value = _convert_to_degrees(gps_longitude.values)
        if gps_longitude_ref.values == 'W':
            lon_value = -lon_value
            
        return (lat_value, lon_value)
    except Exception as e:
        print(f"Error converting GPS data: {e}")
        return None

def _convert_to_degrees(value):
    """Helper function to convert EXIF GPS coordinate format to decimal degrees."""
    d = float(value[0].num) / float(value[0].den)
    m = float(value[1].num) / float(value[1].den)
    s = float(value[2].num) / float(value[2].den)
    return d + (m / 60.0) + (s / 3600.0)

def get_location_from_gps(gps_coords, geolocator=None):
    """Get location name from GPS coordinates using reverse geocoding."""
    if not gps_coords:
        return None
    
    if geolocator is None:
        geolocator = Nominatim(user_agent="memory_cartography_app")
    
    try:
        location = geolocator.reverse(f"{gps_coords[0]}, {gps_coords[1]}")
        
        if not location:
            return None
        
        # Extract city/town and country
        address = location.raw.get('address', {})
        city = (address.get('city') or address.get('town') or 
                address.get('village') or address.get('suburb') or
                address.get('county'))
        country = address.get('country')
        
        if city and country:
            return f"{city}, {country}"
        elif city:
            return city
        else:
            return location.address.split(',')[0]
    except Exception as e:
        print(f"Error reverse geocoding: {e}")
        return None

def get_image_date(exif_data):
    """Extract date taken from EXIF data."""
    if not exif_data:
        return None
    
    # Try different date fields
    for tag in ['EXIF DateTimeOriginal', 'EXIF DateTimeDigitized', 'Image DateTime']:
        date_time = exif_data.get(tag)
        if date_time:
            try:
                # EXIF date format: 'YYYY:MM:DD HH:MM:SS'
                date_str = str(date_time)
                # Convert to ISO format: 'YYYY-MM-DD'
                dt = datetime.strptime(date_str, '%Y:%m:%d %H:%M:%S')
                return dt.strftime('%Y-%m-%d')
            except Exception as e:
                print(f"Error parsing date {date_time}: {e}")
    
    return None

def extract_image_metadata(image_dir, metadata_file=None, use_exif=True):
    """
    Extract metadata from image files in a directory.
    Returns a dict mapping relative image paths to metadata.
    """
    image_dir = Path(image_dir)
    metadata = {}
    
    # Load existing metadata if provided
    if metadata_file and os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    
    # Initialize geocoder for location lookups
    geolocator = Nominatim(user_agent="memory_cartography_app")
    
    # Find all images
    image_paths = list(image_dir.glob("**/*.jpg")) + list(image_dir.glob("**/*.jpeg")) + list(image_dir.glob("**/*.png"))
    
    for img_path in image_paths:
        rel_path = str(img_path.relative_to(image_dir))
        
        # Skip if we already have metadata for this image
        if rel_path in metadata:
            continue
        
        location = None
        date = None
        
        if use_exif:
            # Extract EXIF data
            exif_data = get_exif_data(img_path)
            
            # Get GPS coordinates
            gps_coords = get_gps_data(exif_data)
            
            # Get location name from GPS
            if gps_coords:
                location = get_location_from_gps(gps_coords, geolocator)
            
            # Get date taken
            date = get_image_date(exif_data)
        
        # Fallback to folder name for location if EXIF data not available
        if not location:
            # Try to extract folder name as location
            parent_folder = img_path.parent.name
            if parent_folder and parent_folder != image_dir.name:
                location = parent_folder
            else:
                location = "Unknown"
        
        # Fallback to file modification time for date
        if not date:
            mod_time = os.path.getmtime(img_path)
            date = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d')
        
        metadata[rel_path] = {
            'location': location,
            'date': date
        }
    
    # Save metadata
    if metadata_file:
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    return metadata

def import_images_to_db(db_path, image_dir, metadata, memory_type="user", clip_service=None):
    """Import processed images and metadata to the database."""
    # Convert image_dir to Path object
    image_dir = Path(image_dir)
    
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
    
    # Process each image
    for rel_path, image_meta in metadata.items():
        full_path = image_dir / rel_path
        
        if not os.path.exists(full_path):
            print(f"Warning: Image {full_path} not found, skipping")
            continue
        
        location = image_meta.get('location', 'Unknown')
        date = image_meta.get('date', '2023-01-01')
        
        # Extract keywords and features with CLIP if available
        keywords = []
        embedding = None
        
        if clip_service:
            # Extract keywords
            keyword_results = clip_service.extract_keywords(str(full_path))
            keywords = [kw[0] for kw in keyword_results]
            
            # Extract features
            features = clip_service.encode_image(str(full_path))
            if features is not None:
                embedding = bytes(json.dumps(features.tolist()[0]), 'utf-8')
        
        # Generate a title from location and date
        title = f"{location} - {date}"
        
        # Generate some content from keywords
        content = f"Image taken at {location} on {date}."
        if keywords:
            content += f" Features: {', '.join(keywords[:7])}"
        
        # Check if this image is already in the database
        cursor.execute(
            "SELECT id FROM memories WHERE image_path = ? AND type = ?",
            (rel_path, memory_type)
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
                    location, 
                    date, 
                    json.dumps(keywords), 
                    content,
                    embedding,
                    existing[0]
                )
            )
            print(f"Updated record for {rel_path}")
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
                    location,
                    date,
                    memory_type,
                    json.dumps(keywords),
                    content,
                    embedding,
                    rel_path
                )
            )
            print(f"Added record for {rel_path}")
    
    conn.commit()
    conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process images and prepare database')
    parser.add_argument('--user-dir', type=str, help='Directory containing user images')
    parser.add_argument('--public-dir', type=str, help='Directory containing public images')
    parser.add_argument('--db-path', type=str, default='memories.db', help='Path to SQLite database')
    parser.add_argument('--metadata-file', type=str, help='Path to metadata JSON file')
    parser.add_argument('--no-exif', action='store_true', help='Do not use EXIF data, only folder names')
    
    args = parser.parse_args()
    
    # Initialize CLIP service if available
    clip_service = None
    if get_clip_service:
        clip_service = get_clip_service()
    
    # Process user images
    if args.user_dir:
        print(f"Processing user images from {args.user_dir}...")
        metadata = extract_image_metadata(
            args.user_dir, 
            args.metadata_file, 
            use_exif=not args.no_exif
        )
        import_images_to_db(
            args.db_path, 
            args.user_dir, 
            metadata, 
            "user", 
            clip_service
        )
    
    # Process public images
    if args.public_dir:
        print(f"Processing public images from {args.public_dir}...")
        metadata = extract_image_metadata(
            args.public_dir, 
            args.metadata_file, 
            use_exif=not args.no_exif
        )
        import_images_to_db(
            args.db_path, 
            args.public_dir, 
            metadata, 
            "public", 
            clip_service
        )
    
    print("Done!")