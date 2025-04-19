# memory-map/scripts/process_data.py
import os
import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime
import sqlite3
import exifread
from geopy.geocoders import Nominatim
from PIL import Image
import base64
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv
import torch
import numpy as np

load_dotenv()


class DataProcessor:
    def __init__(self, raw_user_dir, raw_public_dir, processed_user_dir,
                 processed_public_dir, metadata_dir):
        # Set up directories
        self.raw_user_dir = Path(raw_user_dir)
        self.raw_public_dir = Path(raw_public_dir)
        self.processed_user_dir = Path(processed_user_dir)
        self.processed_public_dir = Path(processed_public_dir)
        self.metadata_dir = Path(metadata_dir)

        # Create directories if they don't exist
        os.makedirs(self.processed_user_dir, exist_ok=True)
        os.makedirs(self.processed_public_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)

        # Initialize OpenAI client
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            self.client = OpenAI(api_key=openai_api_key)
        else:
            self.client = None
            print("OPENAI_API_KEY not found in .env file. Keyword extraction will be skipped.")

        # Initialize geocoder
        self.geolocator = Nominatim(user_agent="memory_cartography_app")
        
        # Try to load CLIP model if available (for better embeddings)
        try:
            import clip
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device)
            self.clip_available = True
            print(f"CLIP model loaded successfully on {self.device}")
        except:
            self.clip_available = False
            print("CLIP model not available. Will use OpenAI embeddings instead.")

    def get_exif_data(self, image_path):
        """Extract EXIF data from an image file."""
        try:
            with open(image_path, 'rb') as f:
                return exifread.process_file(f, details=False)
        except Exception as e:
            print(f"Error reading EXIF data from {image_path}: {e}")
            return {}

    def get_gps_coords(self, exif_data):
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
            def convert_to_degrees(value):
                d = float(value[0].num) / float(value[0].den)
                m = float(value[1].num) / float(value[1].den)
                s = float(value[2].num) / float(value[2].den)
                return d + (m / 60.0) + (s / 3600.0)

            lat_value = convert_to_degrees(gps_latitude.values)
            if gps_latitude_ref.values == 'S':
                lat_value = -lat_value

            lon_value = convert_to_degrees(gps_longitude.values)
            if gps_longitude_ref.values == 'W':
                lon_value = -lon_value

            return (lat_value, lon_value)
        except Exception as e:
            print(f"Error converting GPS data: {e}")
            return None

    def get_detailed_location(self, gps_coords):
        """Get detailed location name from GPS coordinates using reverse geocoding."""
        if not gps_coords:
            return None

        try:
            location = self.geolocator.reverse(f"{gps_coords[0]}, {gps_coords[1]}")

            if not location:
                return None

            # Extract detailed address components
            address = location.raw.get('address', {})
            
            # Get specific location elements
            area = address.get('suburb') or address.get('neighbourhood') or address.get('quarter')
            city = address.get('city') or address.get('town') or address.get('village')
            district = address.get('county') or address.get('state_district')
            state = address.get('state') or address.get('region')
            country = address.get('country')

            # Build a detailed location string
            location_parts = []
            if area:
                location_parts.append(area)
            if city:
                location_parts.append(city)
            if district and district not in location_parts:
                location_parts.append(district)
            if state and state not in location_parts:
                location_parts.append(state)
            if country:
                location_parts.append(country)

            if location_parts:
                return ", ".join(location_parts)
            else:
                return location.address.split(',')[0]
        except Exception as e:
            print(f"Error reverse geocoding: {e}")
            return None

    def get_image_date(self, exif_data):
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

    def encode_image(self, image_path):
        """Encodes an image to base64."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            return None

    def extract_keywords_and_sensory(self, image_path):
        """Extract keywords, sensory descriptions, and assign weights using OpenAI API."""
        if not self.client:
            return [], "", 1.0

        base64_image = self.encode_image(image_path)
        if not base64_image:
            return [], "", 1.0

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an assistant that analyzes images and provides detailed descriptions and keywords. For each image, provide:\n\n1. A list of 8-15 keywords including:\n   - Specific items/objects/people in the image\n   - Abstract concepts represented\n   - Sensory qualities (visual, emotional, etc.)\n   - Location-related terms\n\n2. A short sensory description capturing the mood and feeling of the image.\n\n3. A memory importance score from 0.5 to 1.5, where 1.0 is average, higher means more significant/memorable."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Analyze this image and provide keywords, a sensory description, and a memory importance score. Format your response as JSON with fields 'keywords', 'description', and 'weight'."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                        ],
                    }
                ],
                response_format={"type": "json_object"},
                max_tokens=500,
            )

            if response.choices and response.choices[0].message.content:
                try:
                    result = json.loads(response.choices[0].message.content)
                    keywords = result.get('keywords', [])
                    description = result.get('description', "")
                    weight = float(result.get('weight', 1.0))
                    
                    # Ensure all are strings and strip any extra whitespace
                    if isinstance(keywords, str):
                        keywords = [k.strip() for k in keywords.split(',') if k.strip()]
                    
                    return keywords, description, weight
                except json.JSONDecodeError:
                    print(f"Error parsing OpenAI response for {image_path}")
                    return [], "", 1.0
            else:
                print(f"No content received from OpenAI for {image_path}")
                return [], "", 1.0

        except Exception as e:
            print(f"Error calling OpenAI API for {image_path}: {e}")
            return [], "", 1.0

    def generate_embeddings(self, image_path, keywords):
        """Generate embeddings for image and keywords."""
        if self.clip_available:
            try:
                # Generate image embedding using CLIP
                image = Image.open(image_path).convert('RGB')
                image_input = self.preprocess(image).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    image_features = self.clip_model.encode_image(image_input)
                    
                # Generate text embedding for keywords
                text = ", ".join(keywords)
                import clip
                text_tokens = clip.tokenize([text]).to(self.device)
                
                with torch.no_grad():
                    text_features = self.clip_model.encode_text(text_tokens)
                
                # Combine image and text features
                combined_features = (image_features + text_features) / 2
                
                return combined_features.cpu().numpy().tolist()[0]
            except Exception as e:
                print(f"Error generating CLIP embeddings: {e}")
                return []
        elif self.client:
            try:
                # Use OpenAI embedding API as fallback
                text = ", ".join(keywords)
                response = self.client.embeddings.create(
                    input=text,
                    model="text-embedding-ada-002"
                )
                return response.data[0].embedding
            except Exception as e:
                print(f"Error generating OpenAI embeddings: {e}")
                return []
        else:
            return []

    def process_images(self, source_dir, output_dir, prefix, start_idx=1):
        """Process images from source_dir, rename them with prefix, and save to output_dir."""
        # Find all images
        image_paths = []
        for ext in ['.jpg', '.jpeg', '.png']:
            image_paths.extend(list(source_dir.glob(f"**/*{ext}")))
            image_paths.extend(list(source_dir.glob(f"**/*{ext.upper()}")))

        # Sort by name for consistency
        image_paths.sort()

        # Prepare metadata
        metadata = {}

        # Process each image
        idx = start_idx
        for img_path in tqdm(image_paths, desc=f"Processing {prefix} images"):
            try:
                # Generate new filename
                new_filename = f"{prefix}_{idx:04d}{img_path.suffix.lower()}"
                new_path = output_dir / new_filename

                # Extract EXIF data
                exif_data = self.get_exif_data(img_path)

                # Get GPS coordinates
                gps_coords = self.get_gps_coords(exif_data)

                # Get detailed location
                location = None
                if gps_coords:
                    location = self.get_detailed_location(gps_coords)

                # If no GPS data, try to infer from folder name
                if not location:
                    parent_folder = img_path.parent.name
                    if parent_folder and parent_folder != source_dir.name:
                        location = parent_folder.replace('_', ' ')
                    else:
                        # Default location based on prefix
                        location = "Kuala Lumpur, Malaysia" if "KL" in str(img_path) else "Bentong, Malaysia"

                # Get date
                date = self.get_image_date(exif_data)
                if not date:
                    # Use file modification time
                    mod_time = os.path.getmtime(img_path)
                    date = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d')

                # Extract keywords, sensory description, and weight
                keywords, description, weight = self.extract_keywords_and_sensory(img_path)
                
                # If location not in keywords, add it
                location_keywords = [location]
                if ", " in location:
                    location_parts = location.split(", ")
                    for part in location_parts:
                        if part not in location_keywords:
                            location_keywords.append(part)
                
                # Add location to keywords
                for loc in location_keywords:
                    if loc not in keywords:
                        keywords.append(f"{loc} (location)")
                
                # Generate embeddings for similarity search
                embeddings = self.generate_embeddings(img_path, keywords)

                # Copy image to output directory
                shutil.copy2(img_path, new_path)

                # Add to metadata
                metadata[new_filename] = {
                    'original_path': str(img_path),
                    'location': location,
                    'date': date,
                    'keywords': keywords,
                    'description': description,
                    'weight': weight,
                    'embedding': embeddings
                }

                # Increment index
                idx += 1

            except Exception as e:
                print(f"Error processing {img_path}: {e}")

        return metadata

    def create_sqlite_database(self, db_path, user_metadata, public_metadata):
        """Create a SQLite database with metadata."""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Create table
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

        # Insert user metadata
        for filename, data in user_metadata.items():
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
                    'user',
                    json.dumps(data['keywords']),
                    data['description'],
                    data['weight'],
                    json.dumps(data['embedding']) if data['embedding'] else None
                )
            )

        # Insert public metadata
        for filename, data in public_metadata.items():
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
                    'public',
                    json.dumps(data['keywords']),
                    data['description'],
                    data['weight'],
                    json.dumps(data['embedding']) if data['embedding'] else None
                )
            )

        conn.commit()
        conn.close()

    def run(self):
        """Run the complete data processing workflow."""
        # Process user photos
        print("Processing user photos...")
        user_metadata = self.process_images(
            self.raw_user_dir,
            self.processed_user_dir,
            "user",
            start_idx=1
        )

        # Save user metadata
        user_metadata_path = self.metadata_dir / "user_metadata.json"
        with open(user_metadata_path, 'w') as f:
            json.dump(user_metadata, f, indent=2)

        # Process public photos
        print("Processing public photos...")
        public_metadata = self.process_images(
            self.raw_public_dir,
            self.processed_public_dir,
            "public",
            start_idx=1
        )

        # Save public metadata
        public_metadata_path = self.metadata_dir / "public_metadata.json"
        with open(public_metadata_path, 'w') as f:
            json.dump(public_metadata, f, indent=2)

        # Create SQLite database
        print("Creating SQLite database...")
        db_path = self.metadata_dir / "memories.db"
        self.create_sqlite_database(db_path, user_metadata, public_metadata)

        print("Data processing complete!")
        print(f"User photos processed: {len(user_metadata)}")
        print(f"Public photos processed: {len(public_metadata)}")
        print(f"Database created at: {db_path}")
        print(f"User metadata saved to: {user_metadata_path}")
        print(f"Public metadata saved to: {public_metadata_path}")


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