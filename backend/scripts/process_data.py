# memory-map/scripts/process_data.py
import os
import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime
import sqlite3
from PIL import Image
from PIL.ExifTags import TAGS
from geopy.geocoders import Nominatim
import base64
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv
import torch
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('data_processing.log'),
        logging.StreamHandler()
    ]
)
load_dotenv()

class DataProcessor:
    def __init__(self, raw_user_dir, raw_public_dir, processed_user_dir, processed_public_dir, metadata_dir):
        # Set up directories
        self.raw_user_dir = Path(raw_user_dir)
        self.raw_public_dir = Path(raw_public_dir)
        self.processed_user_dir = Path(processed_user_dir)
        self.processed_public_dir = Path(processed_public_dir)
        self.metadata_dir = Path(metadata_dir)
        
        # Create directories if they don't exist
        for dir_path in [self.processed_user_dir, self.processed_public_dir, self.metadata_dir]:
            os.makedirs(dir_path, exist_ok=True)

        # Initialize OpenAI client
        openai_api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=openai_api_key) if openai_api_key else None

        # Initialize geocoder
        self.geolocator = Nominatim(user_agent="memory_cartography_app")

    def find_image_files(self, source_dir):
        """Find image files in the given directory."""
        image_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.gif']
        image_paths = []
        
        for ext in image_extensions:
            # Case-insensitive search, excluding hidden files
            image_paths.extend([
                path for path in source_dir.rglob(f'*{ext.lower()}') 
                if path.is_file() and not path.name.startswith('.')
            ])
            image_paths.extend([
                path for path in source_dir.rglob(f'*{ext.upper()}') 
                if path.is_file() and not path.name.startswith('.')
            ])
        return sorted(set(image_paths))
    
    def extract_image_metadata(self, image_path):
        """Extract comprehensive metadata from an image, including location."""
        try:
            # Read EXIF data using PIL
            img = Image.open(image_path)
            exif_data = {}
            
            if hasattr(img, '_getexif') and img._getexif() is not None:
                exif_data = {
                    TAGS.get(key, key): value
                    for key, value in img._getexif().items()
                }
            
            # Default values
            location = "Unknown Location"
            coordinates = "0.0, 0.0"
            date = datetime.now().strftime('%Y-%m-%d')
            
            # Extract GPS coordinates from EXIF
            if 'GPSInfo' in exif_data:
                try:
                    gps_info = exif_data['GPSInfo']
                    
                    def convert_to_degrees(coordinate):
                        """Convert GPS coordinates from DMS to decimal degrees."""
                        degrees = coordinate[0]
                        minutes = coordinate[1]
                        seconds = coordinate[2]
                        return degrees + (minutes / 60.0) + (seconds / 3600.0)
                    
                    # Check for latitude and longitude
                    if 2 in gps_info and 4 in gps_info:
                        lat = convert_to_degrees(gps_info[2])
                        lat_ref = gps_info.get(1, 'N')
                        lon = convert_to_degrees(gps_info[4])
                        lon_ref = gps_info.get(3, 'E')
                        
                        # Adjust sign based on reference
                        if lat_ref == 'S':
                            lat = -lat
                        if lon_ref == 'W':
                            lon = -lon
                        
                        # Store coordinates
                        coordinates = f"{lat:.6f}, {lon:.6f}"
                        
                        # Use geocoder to get detailed location
                        try:
                            location_info = self.geolocator.reverse(f"{lat}, {lon}")
                            if location_info:
                                address = location_info.raw.get('address', {})
                                
                                # Prioritize specific location details
                                location_parts = []
                                
                                # Preferred order of location details
                                priority_keys = [
                                    'neighbourhood', 'suburb', 'city_district', 
                                    'city', 'town', 'county', 'state'
                                ]
                                
                                for key in priority_keys:
                                    if key in address:
                                        location_parts.append(address[key])
                                
                                # Combine location parts
                                if location_parts:
                                    location = ', '.join(location_parts)
                                else:
                                    location = f"Location near {lat:.4f}, {lon:.4f}"
                        
                        except Exception as geo_err:
                            logging.warning(f"Geocoding failed: {geo_err}")
                            location = f"Location near {coordinates}"
                
                except Exception as gps_err:
                    logging.warning(f"GPS coordinate extraction failed: {gps_err}")
            
            # If location is still unknown, use folder name
            if location == "Unknown Location":
                try:
                    # Get parent folder name and clean it up
                    folder_name = image_path.parent.name
                    
                    # Remove common prefixes and clean up the name
                    folder_name = folder_name.replace('_', ' ').replace('-', ' ').title()
                    
                    # Use the cleaned folder name as location
                    location = folder_name
                except Exception as folder_err:
                    logging.warning(f"Folder name extraction failed: {folder_err}")
            
            # Date extraction
            date_tags = ['DateTimeOriginal', 'DateTime', 'DateTimeDigitized']
            for tag in date_tags:
                if tag in exif_data:
                    try:
                        date_str = str(exif_data[tag])
                        date_obj = datetime.strptime(date_str, '%Y:%m:%d %H:%M:%S')
                        date = date_obj.strftime('%Y-%m-%d')
                        break
                    except Exception:
                        continue
            
            return location, date
        
        except Exception as e:
            logging.error(f"Error extracting metadata for {image_path}: {e}")
            return "Unknown Location", datetime.now().strftime('%Y-%m-%d')
        
    
    def analyze_image_emotional_intensity(self, image_path):
        """Analyze image's emotional intensity using OpenAI."""
        if not self.client:
            logging.warning("OpenAI client not initialized. Skipping emotional analysis.")
            # Return default values with some basic emotional keywords
            return ["memory", "image", "photo"], "A photo from personal collection", 1.0

        try:
            # Encode image
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")

            # Call OpenAI for analysis
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": """Analyze the emotional intensity of this image and return a JSON object. 
                        Provide:
                        1. Emotional keywords as a list of strings
                        2. Emotional description as a detailed paragraph
                        3. Emotional intensity score (0.5-1.5)
                        Focus on mood, color, composition, and implied emotion.
                        Format your response as valid JSON with these exact keys: 'keywords', 'description', 'emotional_intensity_score'."""
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Analyze this image and provide a json response with emotional keywords, description, and emotional_intensity_score."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }
                ],
                response_format={"type": "json_object"},
                max_tokens=300
            )

            # Process response
            try:
                result = json.loads(response.choices[0].message.content)
                
                # Extract data with fallbacks
                keywords = result.get('keywords', [])
                description = result.get('description', "")
                weight = result.get('emotional_intensity_score', 1.0)
                
                # Ensure proper types
                if not isinstance(keywords, list):
                    if isinstance(keywords, str):
                        # Try to convert comma-separated string to list
                        keywords = [k.strip() for k in keywords.split(',')]
                    else:
                        keywords = []
                
                # Clean and validate keywords
                keywords = [k.strip() for k in keywords if k and isinstance(k, str)]
                
                # Ensure we have at least some default keywords if none were provided
                if not keywords:
                    keywords = ["memory", "image", "photo"]
                
                # Ensure description is a string
                if not isinstance(description, str) or not description:
                    description = "A photo from personal collection"
                
                # Ensure weight is a float in the correct range
                try:
                    weight = float(weight)
                    weight = max(0.5, min(1.5, weight))
                except (ValueError, TypeError):
                    weight = 1.0
                
                logging.info(f"Successfully analyzed {image_path}: {len(keywords)} keywords, description length: {len(description)}")
                return keywords, description, weight
            
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse JSON from OpenAI response for {image_path}: {e}")
                logging.debug(f"Response content: {response.choices[0].message.content[:100]}...")
                return ["memory", "image", "photo"], "A photo from personal collection", 1.0
        
        except Exception as e:
            logging.error(f"Emotional analysis failed for {image_path}: {e}")
            return ["memory", "image", "photo"], "A photo from personal collection", 1.0

    def process_images(self, source_dir, output_dir, prefix):
        """Process images from source directory."""
        # Find image files
        image_paths = self.find_image_files(source_dir)
        logging.info(f"Found {len(image_paths)} images in {source_dir}")

        # Prepare metadata dictionary
        metadata = {}

        # Process images
        for idx, img_path in enumerate(tqdm(image_paths, desc=f"Processing {prefix} images"), 1):
            try:
                # Generate new filename
                new_filename = f"{prefix}_{idx:04d}{img_path.suffix.lower()}"
                new_path = output_dir / new_filename

                # Extract location and date
                location, date = self.extract_image_metadata(img_path)

                # Analyze emotional intensity
                keywords, description, weight = self.analyze_image_emotional_intensity(img_path)

                # Add location to keywords
                location_keywords = [f"{location} (location)"]
                keywords.extend(location_keywords)

                # Copy image to processed directory
                shutil.copy2(img_path, new_path)

                # Store metadata
                metadata[new_filename] = {
                    'original_path': str(img_path),
                    'location': location,
                    'date': date,
                    'keywords': keywords,
                    'description': description,
                    'weight': weight
                }

            except Exception as e:
                logging.error(f"Error processing {img_path}: {e}")

        return metadata

    def create_sqlite_database(self, db_path, user_metadata, public_metadata):
        """Create SQLite database with processed metadata."""
        db_exists = os.path.exists(db_path)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        if not db_exists:
            # Create a fresh memories table with embedding field
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
        else:
            # Check if the embedding column exists
            cursor.execute("PRAGMA table_info(memories)")
            columns = [info[1] for info in cursor.fetchall()]
            
            # Add embedding column if it doesn't exist
            if "embedding" not in columns:
                logging.info("Adding embedding column to existing database")
                cursor.execute("ALTER TABLE memories ADD COLUMN embedding TEXT")

        # Helper function to generate simple embedding
        def generate_simple_embedding(data):
            """Generate a simple embedding based on text data."""
            # Combine text data for embedding
            text_data = f"{data.get('location', '')} {data.get('description', '')}"
            if 'keywords' in data and data['keywords']:
                if isinstance(data['keywords'], list):
                    text_data += " " + " ".join(data['keywords'])
            
            # In a real system, you would use a proper embedding model
            # Here we'll just create a simple hash-based vector
            import hashlib
            import struct
            
            # Create a simple 8-dimension embedding (for demonstration purposes)
            md5 = hashlib.md5(text_data.encode('utf-8')).digest()
            floats = [struct.unpack('f', md5[i:i+4])[0] for i in range(0, min(32, len(md5)), 4)]
            
            # Normalize the vector
            magnitude = sum(x**2 for x in floats) ** 0.5
            if magnitude > 0:
                normalized = [x/magnitude for x in floats]
            else:
                normalized = [0.0] * len(floats)
                
            return normalized

        # Helper function to insert metadata
        def insert_metadata(metadata, memory_type):
            for filename, data in metadata.items():
                # Ensure keywords and description exist
                if 'keywords' not in data or not data['keywords']:
                    data['keywords'] = ["memory", "image"]
                
                if 'description' not in data or not data['description']:
                    data['description'] = f"A {memory_type} memory from {data.get('location', 'unknown location')}"
                
                # Generate embedding
                embedding = generate_simple_embedding(data)
                
                cursor.execute(
                    '''
                    INSERT INTO memories
                    (filename, title, location, date, type, keywords, description, weight, embedding)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''',
                    (
                        filename,
                        f"{data['location']} - {data['date']}",
                        data['location'],
                        data['date'],
                        memory_type,
                        json.dumps(data.get('keywords', [])),
                        data.get('description', ''),
                        data.get('weight', 1.0),
                        json.dumps(embedding)
                    )
                )

        # Insert metadata
        insert_metadata(user_metadata, 'user')
        insert_metadata(public_metadata, 'public')

        conn.commit()
        conn.close()
    
    def run(self):
        """Execute complete data processing workflow."""
        try:
            # Process user photos
            logging.info("Processing user photos...")
            user_metadata = self.process_images(
                self.raw_user_dir,
                self.processed_user_dir,
                "user"
            )

            # Save user metadata
            user_metadata_path = self.metadata_dir / "user_metadata.json"
            with open(user_metadata_path, 'w') as f:
                json.dump(user_metadata, f, indent=2)

            # Process public photos
            logging.info("Processing public photos...")
            public_metadata = self.process_images(
                self.raw_public_dir,
                self.processed_public_dir,
                "public"
            )

            # Save public metadata
            public_metadata_path = self.metadata_dir / "public_metadata.json"
            with open(public_metadata_path, 'w') as f:
                json.dump(public_metadata, f, indent=2)

            # Create SQLite database
            logging.info("Creating SQLite database...")
            db_path = self.metadata_dir / "memories.db"
            self.create_sqlite_database(db_path, user_metadata, public_metadata)

            # Final logging
            logging.info(f"User photos processed: {len(user_metadata)}")
            logging.info(f"Public photos processed: {len(public_metadata)}")
            logging.info(f"Database created at: {db_path}")

        except Exception as e:
            logging.error(f"Data processing failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images for Memory Cartography")
    parser.add_argument("--raw-user", default="data/raw/user_photos", help="Directory with raw user photos")
    parser.add_argument("--raw-public", default="data/raw/public_photos", help="Directory with raw public photos")
    parser.add_argument("--processed-user", default="data/processed/user_photos", help="Output directory for processed user photos")
    parser.add_argument("--processed-public", default="data/processed/public_photos", help="Output directory for processed public photos")
    parser.add_argument("--metadata-dir", default="data/metadata", help="Directory for metadata files")

    args = parser.parse_args()

    processor = DataProcessor(
        args.raw_user,
        args.raw_public,
        args.processed_user,
        args.processed_public,
        args.metadata_dir,
    )

    processor.run()