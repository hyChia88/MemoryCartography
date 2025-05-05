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
        """Extract comprehensive metadata from an image."""
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
            date = datetime.now().strftime('%Y-%m-%d')
            
            # Extract location from parent folder if possible
            parent_folder = image_path.parent.name
            if parent_folder and parent_folder != image_path.parent.parent.name:
                location = parent_folder.replace('_', ' ')
            
            # Try to extract date from EXIF
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
                        "content": """You are an expert image analyst tasked with evaluating images related to Pittsburgh, Pennsylvania. Analyze the provided image and return a JSON object.

                        Your analysis should capture both the emotional tone conveyed by the visual elements and the image's potential resonance or significance within the context of Pittsburgh.

                        Provide the following in your JSON response:
                        1.  'keywords': A list of strings (3-5 words) describing the core emotions, moods, or themes evoked (e.g., "nostalgia", "industrial pride", "urban decay", "community", "serenity", "tension").
                        2.  'description': A detailed paragraph explaining the emotional atmosphere. Analyze how mood, color, lighting, composition, and subject matter (including any recognizable Pittsburgh elements) contribute to this atmosphere.
                        3.  'pittsburgh_impact_score': A single floating-point number between 0.2 (minimal emotional impact and low relevance to Pittsburgh) and 2.0 (strong emotional impact and high significance or recognizability related to Pittsburgh). This score should synthesize the visual emotional intensity with the image's connection to the city's identity, landmarks, or culture. A technically well-composed but generic image might score lower than a less polished but emotionally resonant image of a key Pittsburgh scene.

                        Format your entire response as a single, valid JSON object with exactly these keys: "keywords", "description", "pittsburgh_impact_score". Do not include any text outside the JSON object.
                        """
                    },
                    {
                        "role": "user",
                        "content": [
                            # Minimal text needed, system prompt has the details.
                            {"type": "text", "text": "Analyze this image in the context of Pittsburgh according to your instructions."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }
                ],
                response_format={"type": "json_object"},
                max_tokens=400 # Increased slightly for potentially more detailed descriptions
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
    parser.add_argument("--raw-user", default="data/raw/user_photos_ext", help="Directory with raw user photos")
    parser.add_argument("--raw-public", default="data/raw/public_photos_ext", help="Directory with raw public photos")
    parser.add_argument("--processed-user", default="data/processed-ext/user_photos", help="Output directory for processed user photos")
    parser.add_argument("--processed-public", default="data/processed-ext/public_photos", help="Output directory for processed public photos")
    parser.add_argument("--metadata-dir", default="data/metadata-ext", help="Directory for metadata files")

    args = parser.parse_args()

    processor = DataProcessor(
        args.raw_user,
        args.raw_public,
        args.processed_user,
        args.processed_public,
        args.metadata_dir,
    )

    processor.run()