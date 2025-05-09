# app/services/process_data.py
import os
import json
import shutil
import logging
from pathlib import Path
from datetime import datetime
import sqlite3
from PIL import Image
from PIL.ExifTags import TAGS
import base64
import numpy as np
import cv2
import os
import logging

# For ResNet
try:
    import torch
    import torchvision.models as models
    import torchvision.transforms as T
    RESNET_AVAILABLE = True
except ImportError:
    RESNET_AVAILABLE = False
    logging.warning("ResNet dependencies not available. Visual feature extraction will be limited.")

# For YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logging.warning("YOLO not available. Object detection features will be disabled.")

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('logs/data_processing.log', mode='a'),
        logging.StreamHandler()
    ]
)

class DataProcessor:
    """Processes images for memory cartography application."""
    
    def __init__(self, raw_user_dir, raw_public_dir, processed_user_dir, processed_public_dir, metadata_dir, db_path=None, yolo_model_name='yolov8n.pt'):
        """
        Initialize the data processor.
        
        Args:
            raw_user_dir: Directory with original user photos
            raw_public_dir: Directory with original public photos
            processed_user_dir: Output directory for processed user photos
            processed_public_dir: Output directory for processed public photos
            metadata_dir: Directory for metadata
            db_path: Path to SQLite database (if not provided, will use metadata_dir/memories.db)
            yolo_model_name: YOLO model to use for object detection
        """
        # Set up directories
        self.raw_user_dir = Path(raw_user_dir)
        self.raw_public_dir = Path(raw_public_dir)
        self.processed_user_dir = Path(processed_user_dir)
        self.processed_public_dir = Path(processed_public_dir)
        self.metadata_dir = Path(metadata_dir)
        
        # Set database path
        if db_path:
            self.db_path = db_path
        else:
            self.db_path = self.metadata_dir / "memories.db"

        # Ensure directories exist
        for dir_path in [self.processed_user_dir, self.processed_public_dir, self.metadata_dir]:
            os.makedirs(dir_path, exist_ok=True)

        # Initialize models
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if RESNET_AVAILABLE else None
        self._initialize_models(yolo_model_name)
        
        # For storing extracted locations
        self.locations = set()

    def _initialize_models(self, yolo_model_name):
        """Initialize ResNet and YOLO models if available."""
        # Initialize ResNet50
        if RESNET_AVAILABLE:
            try:
                self.resnet_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
                self.resnet_model.eval()  # Set to evaluation mode
                if self.device:
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
        else:
            self.resnet_model = None

        # Initialize YOLO
        if YOLO_AVAILABLE:
            try:
                self.yolo_model = YOLO(yolo_model_name)
                if self.device:
                    self.yolo_model.to(self.device)
                logging.info(f"YOLO model ({yolo_model_name}) initialized successfully.")
            except Exception as e:
                logging.error(f"Failed to initialize YOLO model ({yolo_model_name}): {e}")
                self.yolo_model = None
        else:
            self.yolo_model = None

    def find_image_files(self, source_dir):
        """Find all image files in a directory."""
        image_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.gif']
        image_paths = []
        
        try:
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
        except Exception as e:
            logging.error(f"Error finding images in {source_dir}: {e}")
            return []

    def extract_image_metadata(self, image_path):
        """Extract location and date from image EXIF data or folder structure."""
        try:
            img = Image.open(image_path)
            exif_data = {}
            if hasattr(img, '_getexif') and img._getexif() is not None:
                exif_data = {
                    TAGS.get(key, key): value
                    for key, value in img._getexif().items()
                }

            # Try to determine location from parent folder name
            location = "Unknown Location"
            parent_folder = image_path.parent.name
            if parent_folder and parent_folder != image_path.parent.parent.name:
                location = parent_folder.replace('_', ' ').title()
                # Add to locations set for session
                self.locations.add(location)

            # Try to determine date from EXIF data
            date = datetime.now().strftime('%Y-%m-%d')  # Default to current date
            date_tags = ['DateTimeOriginal', 'DateTime', 'DateTimeDigitized']
            for tag in date_tags:
                if tag in exif_data:
                    try:
                        date_str = str(exif_data[tag])
                        if isinstance(date_str, tuple):
                            date_str = date_str[0]  # If multiple dates, use first
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
        """
        Basic placeholder for emotional analysis.
        In a production app, this would use OpenAI API or similar for analysis.
        
        Returns:
            tuple: (keywords, description, weight)
        """
        # In this privacy-focused version, we'll use simple default values
        # You can add more sophisticated analysis as needed
        filename = os.path.basename(image_path)
        file_parts = filename.split('_')
        
        # Try to extract meaningful keywords from filename
        keywords = []
        if len(file_parts) > 1:
            # Add parts of filename as keywords
            keywords = [part.lower() for part in file_parts if len(part) > 3]
        
        # Add parent folder as a keyword if available
        parent_folder = os.path.basename(os.path.dirname(image_path))
        if parent_folder and len(parent_folder) > 3:
            keywords.append(parent_folder.replace('_', ' ').lower())
            
        # Add default keywords if none extracted
        if not keywords:
            keywords = ["memory", "image", "photo"]
            
        # Simple description based on filename and folder
        description = f"Photo taken at {parent_folder.replace('_', ' ')} location."
        
        # Default weight
        weight = 1.0
            
        return keywords, description, weight

    def extract_resnet_features(self, image_path):
        """Extract visual features using ResNet model."""
        if not RESNET_AVAILABLE or not self.resnet_model:
            return None
            
        try:
            img = Image.open(image_path).convert('RGB')
            transformed_img = self.resnet_transform(img).unsqueeze(0)
            if self.device:
                transformed_img = transformed_img.to(self.device)
                
            with torch.no_grad():
                # Use ResNet model without final fully connected layer
                feature_extractor = torch.nn.Sequential(*list(self.resnet_model.children())[:-1])
                feature_extractor.eval()
                if self.device:
                    feature_extractor.to(self.device)
                    
                features = feature_extractor(transformed_img)  # Shape: (1, 2048, 1, 1)
                
                # Flatten and convert to list
                return features.squeeze().cpu().numpy().tolist()  # Flatten to (2048,)
        except Exception as e:
            logging.error(f"ResNet feature extraction failed for {image_path}: {e}")
            return None

    def detect_objects_yolo(self, image_path):
        """Detect objects in an image using YOLO model."""
        if not YOLO_AVAILABLE or not self.yolo_model:
            return []
            
        try:
            # Perform detection
            results = self.yolo_model(image_path, verbose=False)
            
            detected_objects_info = []
            if results and results[0].boxes:
                boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes (x1, y1, x2, y2)
                confs = results[0].boxes.conf.cpu().numpy()  # Confidences
                clss = results[0].boxes.cls.cpu().numpy()    # Class IDs
                names = results[0].names  # Class names mapping
                
                for i in range(len(boxes)):
                    # Only include objects with confidence above threshold
                    if confs[i] > 0.25:
                        detected_objects_info.append({
                            "box": boxes[i].tolist(),
                            "confidence": float(confs[i]),
                            "class_id": int(clss[i]),
                            "class_name": names[int(clss[i])]
                        })
            
            # Return just the class names for simplicity
            return [obj["class_name"] for obj in detected_objects_info]
        except Exception as e:
            logging.error(f"YOLO object detection failed for {image_path}: {e}")
            return []

    def annotate_image_with_detections(self, original_image_path, detections, output_path):
        """Create an annotated image with object detection results."""
        if not detections:
            # Just copy the original image if no annotations are to be made
            shutil.copy2(original_image_path, output_path)
            return

        try:
            img_cv = cv2.imread(str(original_image_path))
            if img_cv is None:
                logging.error(f"Failed to read image {original_image_path} with OpenCV.")
                shutil.copy2(original_image_path, output_path)  # Fallback to copy
                return

            # Process each detection
            for detection in detections:
                if isinstance(detection, dict) and "box" in detection and "class_name" in detection:
                    # For dict format with full detection info
                    box = detection["box"]
                    label = f"{detection['class_name']}: {detection['confidence']:.2f}"
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
        """
        Process images from source directory and save to output directory.
        
        Args:
            source_dir: Directory with original images
            output_dir: Directory to save processed images
            prefix: Prefix for processed image filenames ('user' or 'public')
            
        Returns:
            dict: Dictionary with metadata for processed images
        """
        image_paths = self.find_image_files(source_dir)
        logging.info(f"Found {len(image_paths)} images in {source_dir}")
        metadata = {}

        for idx, img_path in enumerate(image_paths, 1):
            try:
                new_filename_stem = f"{prefix}_{idx:04d}"
                
                # Ensure the suffix is correct
                original_suffix = img_path.suffix.lower()
                if original_suffix == '.jpeg':  # Standardize to .jpg
                    new_suffix = '.jpg'
                elif original_suffix not in ['.jpg', '.png', '.webp']:  # Fallback for other types
                     new_suffix = '.jpg'  # Convert to jpg if uncommon type
                else:
                    new_suffix = original_suffix

                new_filename = f"{new_filename_stem}{new_suffix}"
                annotated_filename = f"{new_filename_stem}_annotated{new_suffix}"
                
                processed_image_path = output_dir / new_filename
                annotated_image_path = output_dir / annotated_filename

                # 1. Extract EXIF metadata
                location, date = self.extract_image_metadata(img_path)
                
                # 2. Analyze emotional intensity (placeholder in privacy-focused version)
                keywords, description, weight = self.analyze_image_emotional_intensity(img_path)

                # 3. Extract ResNet features (visual embeddings)
                resnet_embedding = self.extract_resnet_features(img_path)

                # 4. Perform YOLO object detection
                detected_objects = self.detect_objects_yolo(img_path)
                
                # 5. Copy original to processed directory
                shutil.copy2(img_path, processed_image_path)
                
                # 6. If objects were detected, create an annotated version
                if detected_objects:
                    # Convert simple list to the format expected by annotate_image_with_detections
                    detection_objs = [
                        {"box": [10, 10, 100, 100], "class_name": obj, "confidence": 0.9}
                        for obj in detected_objects
                    ]
                    self.annotate_image_with_detections(img_path, detection_objs, annotated_image_path)
                else:
                    # No objects detected, just copy the original
                    shutil.copy2(img_path, annotated_image_path)
                
                # Add location to keywords if not already present
                if location != "Unknown Location" and location.lower() not in " ".join(keywords).lower():
                    keywords.append(f"{location.split(',')[0]} (location)")
                
                # Store metadata for this image
                metadata[new_filename] = {
                    'original_path': str(img_path),
                    'processed_path': str(processed_image_path),
                    'annotated_path': str(annotated_image_path),
                    'location': location,
                    'date': date,
                    'openai_keywords': keywords,  # Using our basic keywords in place of OpenAI
                    'openai_description': description,  # Similarly for description
                    'impact_weight': weight,
                    'resnet_embedding': resnet_embedding,
                    'detected_objects': detected_objects
                }
                
                logging.info(f"Processed {img_path} → {processed_image_path}")
                
            except Exception as e:
                logging.error(f"Error processing {img_path}: {e}")
        
        return metadata

    def create_sqlite_database(self, db_path, user_metadata, public_metadata):
        """
        Create or update SQLite database with processed image metadata.
        
        Args:
            db_path: Path to SQLite database
            user_metadata: Dictionary with user image metadata
            public_metadata: Dictionary with public image metadata
        """
        db_exists = os.path.exists(db_path)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        if not db_exists:
            logging.info(f"Creating new database schema in {db_path}")
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
                embedding TEXT,
                openai_keywords TEXT,
                openai_description TEXT,
                impact_weight REAL DEFAULT 1.0,
                resnet_embedding TEXT,
                detected_objects TEXT
            )
            ''')
        else:
            logging.info(f"Using existing database at {db_path}")
            # Check and add new columns if they don't exist
            existing_columns = [info[1] for info in cursor.execute("PRAGMA table_info(memories)").fetchall()]
            
            for column_name, column_type in [
                ("openai_keywords", "TEXT"),
                ("openai_description", "TEXT"),
                ("impact_weight", "REAL DEFAULT 1.0"),
                ("resnet_embedding", "TEXT"),
                ("detected_objects", "TEXT")
            ]:
                if column_name not in existing_columns:
                    cursor.execute(f"ALTER TABLE memories ADD COLUMN {column_name} {column_type}")

        def insert_metadata(metadata_dict, memory_type):
            """Insert metadata into database."""
            for filename, data in metadata_dict.items():
                try:
                    # Check if this filename already exists
                    cursor.execute("SELECT id FROM memories WHERE filename = ?", (filename,))
                    existing_id = cursor.fetchone()
                    
                    if existing_id:
                        # Update existing record
                        cursor.execute(
                            '''
                            UPDATE memories SET
                            title = ?, location = ?, date = ?, type = ?,
                            keywords = ?, description = ?, weight = ?,
                            openai_keywords = ?, openai_description = ?, impact_weight = ?,
                            resnet_embedding = ?, detected_objects = ?
                            WHERE filename = ?
                            ''',
                            (
                                f"{data.get('location', 'Unknown')} - {data.get('date', 'Unknown')}",
                                data.get('location', 'Unknown Location'),
                                data.get('date', datetime.now().strftime('%Y-%m-%d')),
                                memory_type,
                                json.dumps(data.get('openai_keywords', [])),
                                data.get('openai_description', ''),
                                data.get('weight', 1.0),
                                json.dumps(data.get('openai_keywords', [])),
                                data.get('openai_description', ''),
                                data.get('impact_weight', 1.0),
                                json.dumps(data.get('resnet_embedding')),
                                json.dumps(data.get('detected_objects', [])),
                                filename
                            )
                        )
                    else:
                        # Insert new record
                        cursor.execute(
                            '''
                            INSERT INTO memories
                            (filename, title, location, date, type, keywords, description, weight,
                            openai_keywords, openai_description, impact_weight, resnet_embedding, detected_objects)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ''',
                            (
                                filename,
                                f"{data.get('location', 'Unknown')} - {data.get('date', 'Unknown')}",
                                data.get('location', 'Unknown Location'),
                                data.get('date', datetime.now().strftime('%Y-%m-%d')),
                                memory_type,
                                json.dumps(data.get('openai_keywords', [])),
                                data.get('openai_description', ''),
                                data.get('weight', 1.0),
                                json.dumps(data.get('openai_keywords', [])),
                                data.get('openai_description', ''),
                                data.get('impact_weight', 1.0),
                                json.dumps(data.get('resnet_embedding')),
                                json.dumps(data.get('detected_objects', []))
                            )
                        )
                except sqlite3.IntegrityError as e:
                    logging.warning(f"Database integrity error for {filename}: {e}")
                except Exception as e:
                    logging.error(f"Error inserting metadata for {filename}: {e}")

        # Insert metadata into database
        if user_metadata:
            insert_metadata(user_metadata, 'user')
            
        if public_metadata:
            insert_metadata(public_metadata, 'public')

        conn.commit()
        conn.close()
        logging.info(f"Database {db_path} updated successfully.")

    def run(self):
        """
        Run the complete data processing workflow.
        """
        try:
            logging.info("Starting data processing workflow...")

            user_metadata = self.process_images(
                self.raw_user_dir,
                self.processed_user_dir,
                "user"
            )
            
            # Save user metadata to JSON file
            user_metadata_path = self.metadata_dir / "user_image_metadata.json"
            with open(user_metadata_path, 'w') as f:
                json.dump(user_metadata, f, indent=2)
            logging.info(f"User metadata saved to {user_metadata_path}")

            public_metadata = self.process_images(
                self.raw_public_dir,
                self.processed_public_dir,
                "public"
            )
            
            # Save public metadata to JSON file
            public_metadata_path = self.metadata_dir / "public_image_metadata.json"
            with open(public_metadata_path, 'w') as f:
                json.dump(public_metadata, f, indent=2)
            logging.info(f"Public metadata saved to {public_metadata_path}")

            # Update database with all metadata
            self.create_sqlite_database(
                self.db_path,
                user_metadata,
                public_metadata
            )

            logging.info(f"Data processing workflow completed. User photos processed: {len(user_metadata)}. Public photos processed: {len(public_metadata)}.")
            
            # Return a summary of processed data
            return {
                "user_photos": len(user_metadata),
                "public_photos": len(public_metadata),
                "locations": list(self.locations)
            }

        except Exception as e:
            logging.error(f"Data processing workflow failed: {e}")
            return {
                "error": str(e),
                "user_photos": 0,
                "public_photos": 0,
                "locations": []
            }