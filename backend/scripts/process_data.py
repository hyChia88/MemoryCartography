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

    def get_location_from_gps(self, gps_coords):
        """Get location name from GPS coordinates using reverse geocoding."""
        if not gps_coords:
            return None

        try:
            location = self.geolocator.reverse(f"{gps_coords[0]}, {gps_coords[1]}")

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

    def extract_keywords(self, image_path):
        """Extract keywords/labels from an image using OpenAI API."""
        if not self.client:
            return []

        base64_image = self.encode_image(image_path)
        if not base64_image:
            return []

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Identify the key objects, scenes, and characteristics in this image. Provide a list of 5-10 concise keywords or short phrases."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                        ],
                    }
                ],
                max_tokens=100,
            )

            if response.choices and response.choices[0].message.content:
                keywords_str = response.choices[0].message.content.strip()
                keywords = [k.strip() for k in keywords_str.split(',') if k.strip()]
                return keywords[:10]
            else:
                print(f"No keywords received from OpenAI for {image_path}")
                return []

        except Exception as e:
            print(f"Error calling OpenAI API for {image_path}: {e}")
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

                # Get location
                location = None
                if gps_coords:
                    location = self.get_location_from_gps(gps_coords)

                # If no GPS data, try to infer from folder name
                if not location:
                    parent_folder = img_path.parent.name
                    if parent_folder and parent_folder != source_dir.name:
                        location = parent_folder.replace('_', ' ')
                    else:
                        # Default location based on prefix
                        location = "Kuala Lumpur" if "KL" in str(img_path) else "Bentong"

                # Get date
                date = self.get_image_date(exif_data)
                if not date:
                    # Use file modification time
                    mod_time = os.path.getmtime(img_path)
                    date = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d')

                # Extract keywords/labels using OpenAI API
                keywords = self.extract_keywords(img_path)

                # Generate description
                description = f"Image taken at {location}. Features: {', '.join(keywords[:5])}"

                # Copy image to output directory
                shutil.copy2(img_path, new_path)

                # Add to metadata
                metadata[new_filename] = {
                    'original_path': str(img_path),
                    'location': location,
                    'date': date,
                    'keywords': keywords,
                    'description': description
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
            description TEXT
        )
        ''')

        # Insert user metadata
        for filename, data in user_metadata.items():
            cursor.execute(
                '''
                INSERT INTO memories
                (filename, title, location, date, type, keywords, description)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ''',
                (
                    filename,
                    f"{data['location']} - {data['date']}",
                    data['location'],
                    data['date'],
                    'user',
                    json.dumps(data['keywords']),
                    data['description']
                )
            )

        # Insert public metadata
        for filename, data in public_metadata.items():
            cursor.execute(
                '''
                INSERT INTO memories
                (filename, title, location, date, type, keywords, description)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ''',
                (
                    filename,
                    f"{data['location']} - {data['date']}",
                    data['location'],
                    data['date'],
                    'public',
                    json.dumps(data['keywords']),
                    data['description']
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
        print(db_path)
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
    print("done")