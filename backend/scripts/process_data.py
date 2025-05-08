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
import cv2 # Added for image processing and annotations

# For ResNet
import torchvision.models as models
import torchvision.transforms as T

# For YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("WARNING: YOLO not available. Install with 'pip install ultralytics'")
    print("Object detection and annotation features will be disabled.")


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('data_processing.log', mode='w'), # Overwrite log file each run
        logging.StreamHandler()
    ]
)
load_dotenv()

class DataProcessor:
    def __init__(self, raw_user_dir, raw_public_dir, processed_user_dir, processed_public_dir, metadata_dir, yolo_model_name='yolov8n.pt'):
        # Set up directories
        self.raw_user_dir = Path(raw_user_dir)
        self.raw_public_dir = Path(raw_public_dir)
        self.processed_user_dir = Path(processed_user_dir)
        self.processed_public_dir = Path(processed_public_dir)
        self.metadata_dir = Path(metadata_dir)

        for dir_path in [self.processed_user_dir, self.processed_public_dir, self.metadata_dir]:
            os.makedirs(dir_path, exist_ok=True)

        # Initialize OpenAI client
        openai_api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=openai_api_key) if openai_api_key else None
        if not self.client:
            logging.warning("OpenAI API key not found or client failed to initialize. Emotional analysis will be skipped.")

        # Initialize geocoder
        self.geolocator = Nominatim(user_agent="memory_cartography_app")

        # Initialize ResNet and YOLO models
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")
        self._initialize_models(yolo_model_name)

    def _initialize_models(self, yolo_model_name):
        # Initialize ResNet50
        try:
            self.resnet_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            self.resnet_model.eval() # Set to evaluation mode
            self.resnet_model.to(self.device)
            # Define ResNet transformations
            self.resnet_transform = T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            logging.info("ResNet50 model initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize ResNet50 model: {e}")
            self.resnet_model = None

        # Initialize YOLO
        if YOLO_AVAILABLE:
            try:
                self.yolo_model = YOLO(yolo_model_name)
                self.yolo_model.to(self.device)
                logging.info(f"YOLO model ({yolo_model_name}) initialized successfully.")
            except Exception as e:
                logging.error(f"Failed to initialize YOLO model ({yolo_model_name}): {e}")
                self.yolo_model = None
        else:
            self.yolo_model = None
            logging.warning("YOLO model could not be initialized because ultralytics is not available.")

    def find_image_files(self, source_dir):
        image_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.gif']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend([
                path for path in source_dir.rglob(f'*{ext.lower()}')
                if path.is_file() and not path.name.startswith('.')
            ])
            image_paths.extend([
                path for path in source_dir.rglob(f'*{ext.upper()}')
                if path.is_file() and not path.name.startswith('.')
            ])
        return sorted(list(set(image_paths)))


    def extract_image_metadata(self, image_path):
        try:
            img = Image.open(image_path)
            exif_data = {}
            if hasattr(img, '_getexif') and img._getexif() is not None:
                exif_data = {
                    TAGS.get(key, key): value
                    for key, value in img._getexif().items()
                }

            location = "Unknown Location"
            parent_folder = image_path.parent.name
            if parent_folder and parent_folder != image_path.parent.parent.name: # Basic check
                 try:
                    # Attempt to geocode folder name if it seems like a place
                    geo_location = self.geolocator.geocode(parent_folder.replace('_', ' '), timeout=5)
                    if geo_location:
                        location = geo_location.address
                    else:
                        location = parent_folder.replace('_', ' ').title()
                 except Exception: # Geopy timeout or other error
                    location = parent_folder.replace('_', ' ').title()


            date = datetime.now().strftime('%Y-%m-%d') # Default to now
            date_tags = ['DateTimeOriginal', 'DateTime', 'DateTimeDigitized']
            for tag in date_tags:
                if tag in exif_data:
                    try:
                        date_str = str(exif_data[tag])
                        # Handle various possible EXIF date formats
                        if isinstance(date_str, tuple): date_str = date_str[0] # if multiple dates
                        date_obj = datetime.strptime(date_str.split(" ")[0], '%Y:%m:%d')
                        date = date_obj.strftime('%Y-%m-%d')
                        break
                    except (ValueError, TypeError, IndexError):
                        continue
            return location, date
        except Exception as e:
            logging.error(f"Error extracting EXIF metadata for {image_path}: {e}")
            return "Unknown Location", datetime.now().strftime('%Y-%m-%d')

    def analyze_image_emotional_intensity(self, image_path):
        if not self.client:
            logging.warning("OpenAI client not initialized. Skipping emotional analysis.")
            return ["memory", "image", "photo"], "A photo from personal collection", 1.0

        try:
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")
            
            response = self.client.chat.completions.create(
                model="gpt-4o", # or "gpt-4-turbo" / "gpt-4-vision-preview" if needed
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert image analyst. Analyze the provided image.
                        Return a JSON object with:
                        1.  'keywords': A list of 3-5 strings describing core emotions, moods, or themes.
                        2.  'description': A paragraph explaining the emotional atmosphere.
                        3.  'impact_score': A float between 0.2 (minimal impact) and 2.0 (strong impact).
                        Format as a single JSON object: {"keywords": [], "description": "", "impact_score": 1.0}."""
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Analyze this image according to your instructions."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }
                ],
                response_format={"type": "json_object"},
                max_tokens=400
            )
            result = json.loads(response.choices[0].message.content)
            keywords = result.get('keywords', ["memory"])
            description = result.get('description', "A personal photo.")
            # Ensure impact_score is used for weight, previously emotional_intensity_score
            weight = float(result.get('impact_score', 1.0))
            weight = max(0.2, min(2.0, weight)) # Clamp to expected range
            
            return keywords, description, weight
        except Exception as e:
            logging.error(f"OpenAI Emotional analysis failed for {image_path}: {e}")
            return ["error", "analysis_failed"], "Analysis failed.", 1.0

    def extract_resnet_features(self, image_path):
        if not self.resnet_model:
            logging.warning("ResNet model not available. Skipping feature extraction.")
            return None
        try:
            img = Image.open(image_path).convert('RGB')
            transformed_img = self.resnet_transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = self.resnet_model(transformed_img)
            # Using the output of the average pooling layer directly as features
            # For ResNet50, this is a 2048-dimensional vector
            # We can get it by registering a forward hook or by removing the fc layer
            # A simpler way for ResNet50 is to use a hook to grab features before the final FC layer
            
            # Alternative: remove final layer
            feature_extractor = torch.nn.Sequential(*list(self.resnet_model.children())[:-1]) # Remove fc
            feature_extractor.eval()
            feature_extractor.to(self.device)
            with torch.no_grad():
                features = feature_extractor(transformed_img) # Shape: (1, 2048, 1, 1)
            
            # Flatten and convert to list
            return features.squeeze().cpu().numpy().tolist() # Flatten to (2048,)
        except Exception as e:
            logging.error(f"ResNet feature extraction failed for {image_path}: {e}")
            return None

    def detect_objects_yolo(self, image_path):
        if not self.yolo_model:
            logging.warning("YOLO model not available. Skipping object detection.")
            return [], None # Return empty list for objects and None for annotated image data
        try:
            # Perform detection
            results = self.yolo_model(image_path, verbose=False) # verbose=False to reduce console output
            
            detected_objects_info = []
            if results and results[0].boxes:
                boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes (x1, y1, x2, y2)
                confs = results[0].boxes.conf.cpu().numpy()  # Confidences
                clss = results[0].boxes.cls.cpu().numpy()    # Class IDs
                names = results[0].names # Class names mapping
                
                for i in range(len(boxes)):
                    detected_objects_info.append({
                        "box": boxes[i].tolist(),
                        "confidence": float(confs[i]),
                        "class_id": int(clss[i]),
                        "class_name": names[int(clss[i])]
                    })
            return detected_objects_info
        except Exception as e:
            logging.error(f"YOLO object detection failed for {image_path}: {e}")
            return [], None

    def annotate_image_with_detections(self, original_image_path, detections, output_path):
        if not detections: # No detections or YOLO failed
            # Just copy the original image if no annotations are to be made
            shutil.copy2(original_image_path, output_path)
            return

        try:
            img_cv = cv2.imread(str(original_image_path))
            if img_cv is None:
                logging.error(f"Failed to read image {original_image_path} with OpenCV.")
                shutil.copy2(original_image_path, output_path) # Fallback to copy
                return

            for det in detections:
                box = det["box"]
                label = f"{det['class_name']}: {det['confidence']:.2f}"
                x1, y1, x2, y2 = map(int, box)

                # Draw bounding box
                cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Put label
                cv2.putText(img_cv, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.imwrite(str(output_path), img_cv)
            logging.info(f"Annotated image saved to {output_path}")

        except Exception as e:
            logging.error(f"Failed to annotate image {original_image_path}: {e}. Copying original instead.")
            shutil.copy2(original_image_path, output_path)


    def process_images(self, source_dir, output_dir, prefix):
        image_paths = self.find_image_files(source_dir)
        logging.info(f"Found {len(image_paths)} images in {source_dir}")
        metadata = {}

        for idx, img_path in enumerate(tqdm(image_paths, desc=f"Processing {prefix} images"), 1):
            try:
                new_filename_stem = f"{prefix}_{idx:04d}"
                # Ensure the suffix is correct, handle cases like .jpeg -> .jpg for consistency if desired
                original_suffix = img_path.suffix.lower()
                if original_suffix == '.jpeg': # Standardize to .jpg
                    new_suffix = '.jpg'
                elif original_suffix not in ['.jpg', '.png', '.webp']: # Fallback for other types
                     new_suffix = '.jpg' # Convert to jpg if uncommon type
                else:
                    new_suffix = original_suffix

                annotated_filename = f"{new_filename_stem}_annotated{new_suffix}"
                annotated_image_path = output_dir / annotated_filename

                # 1. Extract EXIF metadata
                location, date = self.extract_image_metadata(img_path)

                # 2. Analyze emotional intensity with OpenAI
                keywords, description, weight = ["default"], "No description", 1.0 # Defaults
                if self.client: # Only run if OpenAI client is available
                    keywords, description, weight = self.analyze_image_emotional_intensity(img_path)
                else:
                    logging.warning(f"Skipping OpenAI analysis for {img_path} as client is not available.")


                # 3. Extract ResNet features (visual embeddings)
                resnet_embedding = self.extract_resnet_features(img_path)

                # 4. Perform YOLO object detection
                yolo_detections = self.detect_objects_yolo(img_path)
                detected_object_names = [det["class_name"] for det in yolo_detections]

                # 5. Annotate and save the image
                self.annotate_image_with_detections(img_path, yolo_detections, annotated_image_path)
                
                # Add location to keywords from EXIF if not already prominent
                if location != "Unknown Location" and location.lower() not in " ".join(keywords).lower():
                    keywords.append(f"{location.split(',')[0]} (location)") # Add primary location part

                metadata[annotated_filename] = {
                    'original_path': str(img_path),
                    'processed_path': str(annotated_image_path), # Path to the annotated image
                    'location': location,
                    'date': date,
                    'openai_keywords': keywords, # Keywords from OpenAI
                    'openai_description': description,
                    'impact_weight': weight, # From OpenAI impact_score
                    'resnet_embedding': resnet_embedding, # Visual embedding
                    'detected_objects': detected_object_names # List of detected object names
                }
            except Exception as e:
                logging.error(f"Error processing {img_path}: {e}", exc_info=True)
        return metadata

    def create_sqlite_database(self, db_path, user_metadata, public_metadata):
        db_exists = os.path.exists(db_path)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        if not db_exists:
            logging.info(f"Creating new database schema in {db_path}")
            cursor.execute('''
            CREATE TABLE memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL UNIQUE, -- Annotated image filename
                original_path TEXT,
                title TEXT NOT NULL,
                location TEXT NOT NULL,
                date TEXT NOT NULL,
                type TEXT NOT NULL, -- 'user' or 'public'
                openai_keywords TEXT, -- JSON string list from OpenAI
                openai_description TEXT,
                impact_weight REAL DEFAULT 1.0,
                resnet_embedding TEXT, -- JSON string of list/vector
                detected_objects TEXT -- JSON string list of object names
            )
            ''')
        else:
            logging.info(f"Checking existing database schema in {db_path}")
            # Check and add new columns if they don't exist (idempotent)
            existing_columns = [info[1] for info in cursor.execute("PRAGMA table_info(memories)").fetchall()]
            if "original_path" not in existing_columns:
                cursor.execute("ALTER TABLE memories ADD COLUMN original_path TEXT")
            if "openai_keywords" not in existing_columns:
                 cursor.execute("ALTER TABLE memories ADD COLUMN openai_keywords TEXT") # Renamed from 'keywords'
            if "openai_description" not in existing_columns:
                 cursor.execute("ALTER TABLE memories ADD COLUMN openai_description TEXT") # Renamed from 'description'
            if "impact_weight" not in existing_columns:
                 cursor.execute("ALTER TABLE memories ADD COLUMN impact_weight REAL DEFAULT 1.0") # Renamed from 'weight'
            if "resnet_embedding" not in existing_columns:
                cursor.execute("ALTER TABLE memories ADD COLUMN resnet_embedding TEXT")
            if "detected_objects" not in existing_columns:
                cursor.execute("ALTER TABLE memories ADD COLUMN detected_objects TEXT")
            
            # Remove old columns if they were renamed (optional cleanup)
            # Example: if 'keywords' was renamed to 'openai_keywords'
            # For simplicity, this step is often skipped or handled manually if data migration is complex.
            # If 'embedding' was the old text hash, and 'resnet_embedding' is the new one:
            # if "embedding" in existing_columns and "resnet_embedding" in existing_columns:
            #     logging.info("Potential old 'embedding' column found, consider migrating or dropping if replaced by 'resnet_embedding'.")


        def insert_metadata(metadata_dict, memory_type):
            for filename, data in tqdm(metadata_dict.items(), desc=f"Inserting {memory_type} metadata"):
                try:
                    # Use processed_path for filename in DB, as it's the annotated one
                    cursor.execute(
                        '''
                        INSERT OR REPLACE INTO memories 
                        (filename, original_path, title, location, date, type, openai_keywords, openai_description, impact_weight, resnet_embedding, detected_objects)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''',
                        (
                            filename, # Annotated filename
                            data.get('original_path'),
                            f"{data.get('location', 'N/A')} - {data.get('date', 'N/A')}",
                            data.get('location', 'Unknown Location'),
                            data.get('date', 'Unknown Date'),
                            memory_type,
                            json.dumps(data.get('openai_keywords', [])),
                            data.get('openai_description', ''),
                            data.get('impact_weight', 1.0),
                            json.dumps(data.get('resnet_embedding')), # Store as JSON string
                            json.dumps(data.get('detected_objects', [])) # Store as JSON string
                        )
                    )
                except sqlite3.IntegrityError as e:
                    logging.warning(f"Skipping duplicate or integrity error for {filename} ({memory_type}): {e}")
                except Exception as e:
                    logging.error(f"Error inserting metadata for {filename} ({memory_type}): {e}")


        insert_metadata(user_metadata, 'user')
        insert_metadata(public_metadata, 'public')

        conn.commit()
        conn.close()
        logging.info(f"Database {db_path} updated successfully.")

    def run(self):
        try:
            logging.info("Starting data processing workflow...")

            user_metadata = self.process_images(
                self.raw_user_dir,
                self.processed_user_dir,
                "user"
            )
            user_metadata_path = self.metadata_dir / "user_image_metadata_with_features.json"
            with open(user_metadata_path, 'w') as f:
                json.dump(user_metadata, f, indent=2)
            logging.info(f"User metadata with features saved to {user_metadata_path}")

            public_metadata = self.process_images(
                self.raw_public_dir,
                self.processed_public_dir,
                "public"
            )
            public_metadata_path = self.metadata_dir / "public_image_metadata_with_features.json"
            with open(public_metadata_path, 'w') as f:
                json.dump(public_metadata, f, indent=2)
            logging.info(f"Public metadata with features saved to {public_metadata_path}")

            self.create_sqlite_database(
                self.metadata_dir / "memories_with_visual_features.db", # New DB name
                user_metadata,
                public_metadata
            )

            logging.info(f"Data processing workflow completed. User photos processed: {len(user_metadata)}. Public photos processed: {len(public_metadata)}.")

        except Exception as e:
            logging.error(f"Data processing workflow failed: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images for Memory Cartography with advanced feature extraction")
    parser.add_argument("--raw-user", default="data/raw/user_photos_ext", help="Directory with raw user photos")
    parser.add_argument("--raw-public", default="data/raw/public_photos_ext", help="Directory with raw public photos")
    parser.add_argument("--processed-user", default="data/processed_ext/user_photos_annotated", help="Output directory for processed & annotated user photos")
    parser.add_argument("--processed-public", default="data/processed_ext/public_photos_annotated", help="Output directory for processed & annotated public photos")
    parser.add_argument("--metadata-dir", default="data/metadata_ext_featured", help="Directory for metadata files and DB")
    parser.add_argument("--yolo-model", default="yolov8n.pt", help="YOLO model name or path (e.g., yolov8n.pt, yolov8s.pt)")


    args = parser.parse_args()

    # Create directories from args if they don't exist, especially output ones
    Path(args.processed_user).mkdir(parents=True, exist_ok=True)
    Path(args.processed_public).mkdir(parents=True, exist_ok=True)
    Path(args.metadata_dir).mkdir(parents=True, exist_ok=True)

    processor = DataProcessor(
        args.raw_user,
        args.raw_public,
        args.processed_user,
        args.processed_public,
        args.metadata_dir,
        yolo_model_name=args.yolo_model
    )

    processor.run()