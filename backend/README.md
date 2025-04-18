# Step 1: Project Setup
bash# Clone your project repository (if using git)
git clone https://github.com/your-username/memory-map.git
cd memory-map

# Set up directory structure
mkdir -p data/raw/user_photos
mkdir -p data/raw/public_photos
mkdir -p data/processed/user_photos
mkdir -p data/processed/public_photos
mkdir -p data/metadata

# Set up Python environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install torch torchvision clip exifread geopy pillow tqdm numpy

# Step 2: Data Collection
For User Photos:
bash# Manually copy your personal photos to the raw user photos directory
cp /path/to/your/photos/* data/raw/user_photos/
For Public Photos (Using Pexels Scraper):
bash# Create a scraper script (use the Pexels scraper from previous message)
mkdir -p scripts
# Place the pexels_scraper.py in the scripts directory

# Run the scraper to download public photos
python scripts/pexels_scraper.py "Kuala Lumpur" --output data/raw/public_photos --num 100
python scripts/pexels_scraper.py "Bentong" --output data/raw/public_photos --num 100
Step 3: Data Processing
bash# Place the process_data.py script in the scripts directory

# Run the data processing script
python scripts/process_data.py
This will:

Rename all images sequentially (user_0001.jpg, user_0002.jpg, etc.)
Extract location data using EXIF when available or folder names
Generate keywords/labels using CLIP
Create metadata files for both datasets
Create a SQLite database with all the information

Step 4: Verify Output
bash# Check the processed directories
ls -l data/processed/user_photos/
ls -l data/processed/public_photos/

# Check the metadata files
cat data/metadata/user_metadata.json | head -n 20
cat data/metadata/public_metadata.json | head -n 20

# Check the database
sqlite3 data/metadata/memories.db "SELECT * FROM memories LIMIT 5;"
Step 5: Use the Data in Your Application

Update your backend to use the processed data:

python# In your backend code, update the database path
DB_PATH = 'data/metadata/memories.db'

Update the frontend to display images from the processed directories:

javascript// In your frontend code
const userImagesPath = 'data/processed/user_photos/';
const publicImagesPath = 'data/processed/public_photos/';
5. Data Format Overview
After processing, your data will be structured as follows:
Processed Directories:

data/processed/user_photos/: Contains renamed user photos (user_0001.jpg, user_0002.jpg, etc.)
data/processed/public_photos/: Contains renamed public photos (public_0001.jpg, public_0002.jpg, etc.)

Metadata Files:

data/metadata/user_metadata.json: A JSON file mapping user photo filenames to their metadata
data/metadata/public_metadata.json: A JSON file mapping public photo filenames to their metadata

Example metadata structure:
json{
  "user_0001.jpg": {
    "original_path": "data/raw/user_photos/kuala_lumpur_trip_123.jpg",
    "location": "Kuala Lumpur, Malaysia",
    "date": "2023-05-15",
    "keywords": ["city", "skyline", "building", "modern", "night", "lights", "architecture", "urban", "tower", "sky"],
    "description": "Image taken at Kuala Lumpur, Malaysia. Features: city, skyline, building, modern, night"
  },
  ...
}
SQLite Database:

data/metadata/memories.db: A SQLite database with a memories table containing all metadata

Database schema:
CREATE TABLE memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    title TEXT NOT NULL,
    location TEXT NOT NULL,
    date TEXT NOT NULL,
    type TEXT NOT NULL,
    keywords TEXT,
    description TEXT
);