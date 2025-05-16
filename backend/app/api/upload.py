# Standard Library Imports
import os
import json
import shutil
import logging
import base64
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

import requests
import asyncio
import aiohttp

# Third-Party Imports
# FastAPI and related
# Add BackgroundTasks back
from fastapi import APIRouter, HTTPException, File, UploadFile, Query, Depends, BackgroundTasks
# from app.core.session import get_session_manager
from pydantic import BaseModel, Field

# File Handling
import aiofiles

# Image Processing & ML
from PIL import Image
from PIL.ExifTags import TAGS
import cv2 # OpenCV for image annotation
import torch
import torchvision.models as models
import torchvision.transforms as T
import numpy as np
from tqdm import tqdm # Progress bars

# Geolocation
from geopy.geocoders import Nominatim

# OpenAI
from openai import OpenAI
from dotenv import load_dotenv

# YOLO (Optional Object Detection)
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    logging.info("Ultralytics YOLO found and available.")
except ImportError:
    YOLO_AVAILABLE = False
    logging.warning("Ultralytics (YOLO) not found. Install with 'pip install ultralytics'. Object detection features will be disabled.")

# --- Application-Specific Imports ---
# Adjust these paths based on your project structure
try:
    from app.core.session import get_session_manager, SessionManager
except ImportError:
    logging.error("Failed to import session manager from app.core.session. Using dummy.")
    # Dummy Session Manager for structure - Replace with your actual implementation
    class SessionManager:
        def get_session_paths(self, session_id: str) -> Optional[Dict[str, str]]: return None
        def add_location(self, session_id: str, location: str): pass
    _dummy_session_manager = SessionManager()
    def get_session_manager(): return _dummy_session_manager

# --- Configuration ---
load_dotenv()
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('logs/upload_processing.log', mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
router = APIRouter()

# --- DataProcessor Class ---
# (Keep the full DataProcessor class definition here as before, or import it)
# NOTE: Ensure the DataProcessor class definition from the previous version is included here.
# For brevity in this example, I'll assume it's defined above or imported correctly.
# Placeholder if it's not defined above:
class DataProcessor:
    """Handles image processing, metadata extraction, AI analysis, and database storage."""
    def __init__(self, raw_user_dir, raw_public_dir, processed_user_dir, processed_public_dir, metadata_dir, yolo_model_name='yolov8n.pt'):
        logger.info("Initializing DataProcessor...")
        # Set up directories
        self.raw_user_dir = Path(raw_user_dir)
        self.raw_public_dir = Path(raw_public_dir)
        self.processed_user_dir = Path(processed_user_dir)
        self.processed_public_dir = Path(processed_public_dir)
        self.metadata_dir = Path(metadata_dir)

        # Ensure output directories exist
        for dir_path in [self.processed_user_dir, self.processed_public_dir, self.metadata_dir]:
            os.makedirs(dir_path, exist_ok=True)
            logger.debug(f"Ensured directory exists: {dir_path}")

        # Initialize OpenAI client
        openai_api_key = os.getenv("OPENAI_API_KEY")
        self.client = None
        if openai_api_key:
            try:
                self.client = OpenAI(api_key=openai_api_key)
                logger.info("OpenAI client initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
        else:
            logger.warning("OPENAI_API_KEY not found in environment variables. OpenAI analysis will be skipped.")

        # Initialize geocoder
        try:
            self.geolocator = Nominatim(user_agent="memory_cartography_app_v1") # Use a specific user agent
            logger.info("Nominatim geolocator initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize Nominatim geolocator: {e}")
            self.geolocator = None

        # Initialize ResNet and YOLO models
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device for ML models: {self.device}")
        self.resnet_model = None
        self.resnet_transform = None
        self.yolo_model = None
        self._initialize_models(yolo_model_name)
        logger.info("DataProcessor initialization complete.")

    def _initialize_models(self, yolo_model_name):
        # Initialize ResNet50
        try:
            logger.debug("Initializing ResNet50 model...")
            # Use weights=... instead of pretrained=True for newer torchvision
            self.resnet_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            # Remove the final fully connected layer to get features
            self.resnet_model = torch.nn.Sequential(*list(self.resnet_model.children())[:-1])
            self.resnet_model.eval() # Set to evaluation mode
            self.resnet_model.to(self.device)
            # Define ResNet transformations matching the pre-trained model
            self.resnet_transform = T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            logger.info("ResNet50 model initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize ResNet50 model: {e}", exc_info=True)
            self.resnet_model = None

        # Initialize YOLO
        if YOLO_AVAILABLE:
            try:
                logger.debug(f"Initializing YOLO model ({yolo_model_name})...")
                self.yolo_model = YOLO(yolo_model_name)
                # self.yolo_model.to(self.device) # YOLO typically handles device placement internally
                logger.info(f"YOLO model ({yolo_model_name}) initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize YOLO model ({yolo_model_name}): {e}", exc_info=True)
                self.yolo_model = None
        else:
            self.yolo_model = None
            logger.warning("YOLO model could not be initialized because ultralytics is not available.")

    def find_image_files(self, source_dir: Path) -> List[Path]:
        """Finds all supported image files recursively in a directory."""
        image_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.gif']
        image_paths = []
        logger.info(f"Searching for image files in: {source_dir}")
        if not source_dir.is_dir():
            logger.warning(f"Source directory does not exist or is not a directory: {source_dir}")
            return []
        for ext in image_extensions:
            image_paths.extend(source_dir.rglob(f'*{ext.lower()}'))
            image_paths.extend(source_dir.rglob(f'*{ext.upper()}')) # Include uppercase extensions

        # Filter out hidden files and ensure they are files
        valid_image_paths = [p for p in image_paths if p.is_file() and not p.name.startswith('.')]
        unique_paths = sorted(list(set(valid_image_paths)))
        logger.info(f"Found {len(unique_paths)} unique image files.")
        return unique_paths
    
    def extract_image_metadata(self, image_path: Path) -> tuple[str, str]:
        """Extracts location and date from image EXIF data or uses fallbacks."""
        location = "Unknown Location"
        date = datetime.now().strftime('%Y-%m-%d') # Default to current date

        try:
            logger.info(f"Extracting metadata for: {image_path.name}")
            img = Image.open(image_path)
            exif_data_raw = img._getexif()

            if exif_data_raw:
                exif_data = {TAGS.get(key, key): value for key, value in exif_data_raw.items()}

                # Extract Date
                date_tags = ['DateTimeOriginal', 'DateTime', 'DateTimeDigitized']
                for tag in date_tags:
                    if tag in exif_data:
                        try:
                            date_str = str(exif_data[tag])
                            # Basic parsing, might need refinement for different formats
                            date_obj = datetime.strptime(date_str.split(" ")[0], '%Y:%m:%d')
                            date = date_obj.strftime('%Y-%m-%d')
                            logger.debug(f"Extracted date {date} from tag {tag} for {image_path.name}")
                            break # Use the first valid date found
                        except (ValueError, TypeError, IndexError):
                            continue # Try next tag if parsing fails

                # Extract GPS Location and geocode
                gps_info = exif_data.get('GPSInfo')
                if gps_info:
                    logger.info(f"Found GPS info for {image_path.name}")
                    if self.geolocator:
                        try:
                            # Convert GPS EXIF data to decimal degrees
                            def get_decimal_from_dms(dms, ref):
                                degrees, minutes, seconds = dms
                                decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
                                if ref in ['S', 'W']:
                                    decimal *= -1
                                return decimal

                            lat_dms = gps_info.get(2) # GPSLatitude
                            lat_ref = gps_info.get(1) # GPSLatitudeRef
                            lon_dms = gps_info.get(4) # GPSLongitude
                            lon_ref = gps_info.get(3) # GPSLongitudeRef

                            if lat_dms and lat_ref and lon_dms and lon_ref:
                                latitude = get_decimal_from_dms(lat_dms, lat_ref)
                                longitude = get_decimal_from_dms(lon_dms, lon_ref)
                                logger.debug(f"Extracted GPS coordinates ({latitude}, {longitude}) for {image_path.name}")

                                # Reverse geocode coordinates
                                geo_location = self.geolocator.reverse((latitude, longitude), exactly_one=True, timeout=10)
                                if geo_location:
                                    # Extract city from the full address
                                    location = self._extract_city_from_address(geo_location.address)
                                    logger.info(f"Geocoded location: {geo_location.address}")
                                    logger.info(f"Extracted city: {location} for {image_path.name}")
                                else:
                                    logger.warning(f"Could not reverse geocode coordinates for {image_path.name}")
                            else:
                                logger.debug(f"Incomplete GPSInfo found for {image_path.name}")
                        except Exception as e:
                            logger.error(f"Error processing GPS data or geocoding for {image_path.name}: {e}")
                    else:
                        logger.warning("Geolocator not available - cannot process GPS data")
                else:
                    logger.info(f"No GPS info found in EXIF for {image_path.name}")

                # Fallback: Use parent directory name if location is still unknown
                if location == "Unknown Location":
                    parent_folder = image_path.parent.name
                    # Basic check to avoid using top-level dir name like 'raw_user'
                    if parent_folder not in ['raw_user', 'raw_public', 'user', 'public', 'raw', 'processed']:
                        location = parent_folder.replace('_', ' ').title()
                        logger.debug(f"Using parent folder name as fallback location: {location}")
            else:
                logger.info(f"No EXIF data found for {image_path.name}")

            # Try to extract location from filename patterns if still unknown
            if location == "Unknown Location":
                location = self._extract_location_from_filename(image_path)

        except Exception as e:
            logger.error(f"Error extracting EXIF metadata for {image_path}: {e}", exc_info=True)

        logger.info(f"Final metadata for {image_path.name}: location='{location}', date='{date}'")
        return location, date

    def _extract_city_from_address(self, address: str) -> str:
        """Extract city name from a full geocoded address."""
        if not address:
            return "Unknown Location"
        
        # Split the address by commas
        address_parts = [part.strip() for part in address.split(',')]
        
        # Common patterns for different address formats:
        # Example: "Street, District, City, State, Country"
        # Example: "Street, City, State, Country" 
        # Example: "City, State, Country"
        
        # List of common non-city terms to skip
        non_city_terms = [
            'unnamed road', 'jalan', 'street', 'road', 'avenue', 'boulevard',
            'block', 'lot', 'unit', 'floor', 'building', 'tower', 'plaza',
            'complex', 'mall', 'center', 'centre', 'district', 'zone', 'sector',
            'postcode', 'postal code', 'zip', 'malaysia', 'singapore', 'thailand',
            'indonesia', 'philippines', 'federal territory'
        ]
        
        # Look for city-like parts (usually 2nd to 4th part from the end)
        for i in range(1, min(4, len(address_parts))):
            candidate = address_parts[-i-1].lower()
            
            # Skip if it's too short or contains non-city terms
            if len(candidate) < 3:
                continue
                
            if any(term in candidate for term in non_city_terms):
                continue
                
            # Skip if it looks like a postcode (contains numbers)
            if any(char.isdigit() for char in candidate):
                continue
                
            # This looks like a city
            city_name = address_parts[-i-1].strip()
            
            # Clean up common city name patterns
            if ',' in city_name:
                city_name = city_name.split(',')[0].strip()
                
            return city_name.title()
        
        # If no good candidate found, try to get something meaningful
        # Look for known city patterns in the full address
        known_cities = [
            'kuala lumpur', 'kl', 'petaling jaya', 'pj', 'shah alam', 'subang jaya',
            'klang', 'ampang', 'cheras', 'kepong', 'bangsar', 'mont kiara',
            'singapore', 'johor bahru', 'penang', 'ipoh', 'malacca', 'kota kinabalu'
        ]
        
        address_lower = address.lower()
        for city in known_cities:
            if city in address_lower:
                if city == 'kl':
                    return 'Kuala Lumpur'
                elif city == 'pj':
                    return 'Petaling Jaya'
                else:
                    return city.title()
        
        # Fallback: return the second-to-last part if available, otherwise first meaningful part
        if len(address_parts) >= 2:
            candidate = address_parts[-2].strip()
            # Clean up the candidate
            if any(term in candidate.lower() for term in non_city_terms):
                # Try the first meaningful part instead
                for part in address_parts:
                    part_clean = part.strip().lower()
                    if (len(part_clean) >= 3 and 
                        not any(term in part_clean for term in non_city_terms) and
                        not any(char.isdigit() for char in part_clean)):
                        return part.strip().title()
            return candidate.title()
        
        # Last resort: return first part of address
        return address_parts[0].strip().title() if address_parts else "Unknown Location"

    def _extract_location_from_filename(self, image_path: Path) -> str:
        """Extract location from filename patterns."""
        filename_lower = image_path.stem.lower()
        
        # Common location patterns in filenames - updated to return city names
        location_patterns = {
            'kl': 'Kuala Lumpur',
            'kuala_lumpur': 'Kuala Lumpur',
            'petaling_jaya': 'Petaling Jaya',
            'pj': 'Petaling Jaya',
            'shah_alam': 'Shah Alam',
            'subang': 'Subang Jaya',
            'singapore': 'Singapore',
            'penang': 'Penang',
            'ipoh': 'Ipoh',
            'johor': 'Johor Bahru',
            'malacca': 'Malacca',
            'beach': 'Coastal Area',
            'park': 'City Park',
            'home': 'Residential Area',
            'office': 'Business District',
            'garden': 'Garden Area',
            'city': 'City Center',
            'downtown': 'Downtown',
            'mall': 'Shopping District',
            'restaurant': 'Dining Area',
            'cafe': 'Cafe Area',
            'hotel': 'Hotel District',
            'vacation': 'Tourist Area',
            'trip': 'Travel Destination',
            'holiday': 'Holiday Destination'
        }
        
        for pattern, location_name in location_patterns.items():
            if pattern in filename_lower:
                logger.info(f"Extracted location '{location_name}' from filename pattern '{pattern}'")
                return location_name
        
        # If no pattern matches, return Unknown Location
        return "Unknown Location"

    def analyze_image_emotional_intensity(self, image_path: Path) -> tuple[List[str], str, float]:
        """Analyzes image using OpenAI Vision API for keywords, description, and impact, incorporating YOLO object detection if applicable."""
        if not self.client:
            logger.warning(f"OpenAI client not available. Skipping analysis for {image_path.name}.")
            return ["no_openai"], "OpenAI analysis skipped.", 1.0

        logger.debug(f"Starting OpenAI analysis for {image_path.name}...")
        try:
            # Read image and encode as base64
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")

            # Perform YOLO object detection if available
            detected_objects = []
            if YOLO_AVAILABLE and self.yolo_model:
                results = self.yolo_model(image_path, verbose=False)  # verbose=False reduces console spam
                if results and results[0].boxes:
                    boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes (x1, y1, x2, y2)
                    confs = results[0].boxes.conf.cpu().numpy()  # Confidences
                    clss = results[0].boxes.cls.cpu().numpy()    # Class IDs
                    names = results[0].names  # Class names mapping (dict: id -> name)

                    for i in range(len(boxes)):
                        class_id = int(clss[i])
                        detected_objects.append({
                            "box": boxes[i].tolist(),  # [x1, y1, x2, y2]
                            "confidence": float(confs[i]),
                            "class_id": class_id,
                            "class_name": names[class_id]  # Get name from mapping
                        })

            response = self.client.chat.completions.create(
                model="gpt-4o",  # Or "gpt-4-vision-preview", "gpt-4-turbo"
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert image analyst specializing in emotional context. Analyze the provided image.
                        Return ONLY a valid JSON object with three keys:
                        1. 'keywords': A list of 3-5 single-word strings capturing the core emotions, atmosphere, or significant themes (e.g., ["joyful", "serene", "urban", "nostalgia", "adventure"]).
                        2. 'description': A concise one-sentence description (max 25 words) summarizing the emotional essence or narrative of the scene.
                        3. 'impact_score': A float between 0.2 (low emotional impact, mundane) and 2.0 (high emotional impact, very evocative or intense).
                        Example JSON: {"keywords": ["serene", "nature", "calm"], "description": "A peaceful forest path invites quiet contemplation.", "impact_score": 1.3}"""
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Analyze this image and provide the JSON output as instructed."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }
                ],
                response_format={"type": "json_object"},
                max_tokens=300,  # Adjust as needed
                temperature=0.5  # Control creativity/randomness
            )

            # Extract and validate the result
            result_json = response.choices[0].message.content
            result = json.loads(result_json)

            keywords = result.get('keywords', ["analysis_error"])
            description = result.get('description', "Analysis failed or format error.")
            weight = float(result.get('impact_score', 1.0))
            weight = max(0.2, min(2.0, weight))  # Clamp score to the defined range

            logger.info(f"OpenAI analysis successful for {image_path.name}. Impact: {weight}")
            return keywords, description, weight

        except json.JSONDecodeError as e:
            logger.error(f"OpenAI analysis failed for {image_path.name} - Invalid JSON received: {result_json}. Error: {e}")
            return ["json_error"], "Failed to parse OpenAI response.", 1.0
        except Exception as e:
            logger.error(f"OpenAI analysis failed for {image_path.name}: {e}", exc_info=True)
            return ["api_error"], "OpenAI API call failed.", 1.0

    def extract_resnet_features(self, image_path: Path) -> Optional[List[float]]:
        """Extracts visual features using the pre-trained ResNet model."""
        if not self.resnet_model or not self.resnet_transform:
            logger.warning(f"ResNet model not available. Skipping feature extraction for {image_path.name}.")
            return None
        try:
            logger.debug(f"Extracting ResNet features for {image_path.name}...")
            img = Image.open(image_path).convert('RGB') # Ensure image is RGB
            transformed_img = self.resnet_transform(img).unsqueeze(0).to(self.device) # Add batch dim and send to device

            with torch.no_grad(): # Disable gradient calculation for inference
                features = self.resnet_model(transformed_img) # Forward pass

            # Features are output of the layer before FC, shape (1, 2048, 1, 1) for ResNet50
            # Flatten and convert to list
            embedding = features.squeeze().cpu().numpy().tolist() # Squeeze dims, move to CPU, convert to numpy, then list
            logger.debug(f"ResNet feature extraction successful for {image_path.name}. Embedding size: {len(embedding)}")
            return embedding
        except Exception as e:
            logger.error(f"ResNet feature extraction failed for {image_path.name}: {e}", exc_info=True)
            return None

    def detect_objects_yolo(self, image_path: Path) -> List[Dict[str, Any]]:
        """Detects objects using YOLO model and returns detection info."""
        if not self.yolo_model:
            logger.warning(f"YOLO model not available. Skipping object detection for {image_path.name}.")
            return []

        try:
            logger.debug(f"Running YOLO object detection for {image_path.name}...")
            # Perform detection
            results = self.yolo_model(image_path, verbose=False) # verbose=False reduces console spam

            detected_objects_info = []
            # Process results (structure might vary slightly based on ultralytics version)
            if results and results[0].boxes:
                boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes (x1, y1, x2, y2)
                confs = results[0].boxes.conf.cpu().numpy()  # Confidences
                clss = results[0].boxes.cls.cpu().numpy()    # Class IDs
                names = results[0].names # Class names mapping (dict: id -> name)

                for i in range(len(boxes)):
                    class_id = int(clss[i])
                    detected_objects_info.append({
                        "box": boxes[i].tolist(), # [x1, y1, x2, y2]
                        "confidence": float(confs[i]),
                        "class_id": class_id,
                        "class_name": names[class_id] # Get name from mapping
                    })
                logger.info(f"YOLO detected {len(detected_objects_info)} objects in {image_path.name}.")
            else:
                logger.info(f"No objects detected by YOLO in {image_path.name}.")

            return detected_objects_info
        except Exception as e:
            logger.error(f"YOLO object detection failed for {image_path.name}: {e}", exc_info=True)
            return [] # Return empty list on error

    def annotate_image_with_detections(self, original_image_path: Path, detections: List[Dict[str, Any]], output_path: Path):
        """Draws bounding boxes on the image based on YOLO detections and saves it."""
        if not detections: # No detections or YOLO failed/skipped
            logger.debug(f"No detections to annotate for {original_image_path.name}. Copying original to {output_path}.")
            try:
                shutil.copy2(original_image_path, output_path) # Copy original if no annotations
            except Exception as e:
                logger.error(f"Failed to copy original image {original_image_path} to {output_path}: {e}")
            return

        try:
            logger.debug(f"Annotating image {original_image_path.name} with {len(detections)} detections...")
            # Read image using OpenCV
            img_cv = cv2.imread(str(original_image_path))
            if img_cv is None:
                logger.error(f"Failed to read image {original_image_path} with OpenCV. Cannot annotate.")
                shutil.copy2(original_image_path, output_path) # Fallback to copy
                return

            # Draw boxes and labels
            for det in detections:
                box = det["box"]
                label = f"{det['class_name']}: {det['confidence']:.2f}"
                x1, y1, x2, y2 = map(int, box) # Convert coordinates to integers

                # Draw bounding box (Green color, thickness 2)
                cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Put label text above the box
                cv2.putText(img_cv, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Save the annotated image
            success = cv2.imwrite(str(output_path), img_cv)
            if success:
                logger.info(f"Annotated image saved successfully to {output_path}")
            else:
                 logger.error(f"Failed to save annotated image to {output_path}")
                 # Optionally copy original as fallback if save fails
                 # shutil.copy2(original_image_path, output_path)

        except Exception as e:
            logger.error(f"Failed to annotate image {original_image_path.name}: {e}. Copying original instead.", exc_info=True)
            try:
                shutil.copy2(original_image_path, output_path) # Fallback copy
            except Exception as copy_e:
                logger.error(f"Fallback copy also failed for {original_image_path}: {copy_e}")

    def process_images(self, source_dir: Path, output_dir: Path, prefix: str) -> Dict[str, Dict[str, Any]]:
        """Processes all images in source_dir, extracts metadata/features, saves annotated images, and returns metadata."""
        image_paths = self.find_image_files(source_dir)
        logger.info(f"Starting processing for {len(image_paths)} images from {source_dir} with prefix '{prefix}'...")
        all_metadata = {}

        if not image_paths:
            logger.warning(f"No images found in {source_dir} to process.")
            return {}

        # Use tqdm for progress bar
        for idx, img_path in enumerate(tqdm(image_paths, desc=f"Processing {prefix} images", unit="image")):
            logger.info(f"Processing image {idx+1}/{len(image_paths)}: {img_path.name}")
            try:
                # Define new filename for the processed (annotated) image
                new_filename_stem = f"{prefix}_{idx+1:04d}" # e.g., user_0001
                original_suffix = img_path.suffix.lower()
                # Standardize to .jpg for consistency, or keep original if common type
                if original_suffix == '.jpeg':
                    new_suffix = '.jpg'
                elif original_suffix not in ['.jpg', '.png', '.webp']:
                    logger.warning(f"Image {img_path.name} has uncommon suffix {original_suffix}. Will attempt to save as .jpg.")
                    new_suffix = '.jpg'
                else:
                    new_suffix = original_suffix

                annotated_filename = f"{new_filename_stem}_annotated{new_suffix}"
                annotated_image_path = output_dir / annotated_filename

                # --- Perform processing steps ---
                # 1. Extract EXIF metadata (Location, Date)
                location, date = self.extract_image_metadata(img_path)

                # 2. Analyze emotional intensity with OpenAI (if available)
                keywords, description, weight = ["default"], "No description", 1.0
                if self.client:
                    keywords, description, weight = self.analyze_image_emotional_intensity(img_path)

                # 3. Extract ResNet features (visual embeddings)
                resnet_embedding = self.extract_resnet_features(img_path)

                # 4. Perform YOLO object detection (if available)
                yolo_detections = self.detect_objects_yolo(img_path)
                detected_object_names = [det["class_name"] for det in yolo_detections] # Get just the names

                # 5. Annotate and save the image (copies if no detections)
                self.annotate_image_with_detections(img_path, yolo_detections, annotated_image_path)

                # --- Assemble Metadata ---
                # Add primary location part to keywords if relevant and not already covered
                if location != "Unknown Location":
                    primary_location = location.split(',')[0].strip()
                    # Check if location is implicitly covered by keywords/description
                    loc_in_keywords = any(primary_location.lower() in k.lower() for k in keywords)
                    loc_in_desc = primary_location.lower() in description.lower()
                    if not loc_in_keywords and not loc_in_desc:
                        keywords.append(f"{primary_location} (location)")
                        logger.debug(f"Added location '{primary_location}' to keywords for {img_path.name}")


                # Store metadata using the ANNOTATED filename as the key
                # Construct relative path for processed_path based on a common root (e.g., session dir parent)
                # This assumes metadata_dir is inside the session directory structure
                try:
                    base_path_for_relative = self.metadata_dir.parent # Assumes metadata is one level down
                    relative_processed_path = str(annotated_image_path.relative_to(base_path_for_relative))
                except ValueError:
                    logger.warning(f"Could not make processed path relative: {annotated_image_path}. Storing absolute path.")
                    relative_processed_path = str(annotated_image_path)


                metadata_entry = {
                    'original_path': str(img_path), # Keep track of the original file
                    'processed_path': relative_processed_path, # Relative path for frontend?
                    'location': location,
                    'date': date,
                    'openai_keywords': keywords,
                    'openai_description': description,
                    'impact_weight': weight,
                    'resnet_embedding': resnet_embedding, # Can be None if failed
                    'detected_objects': detected_object_names # List of strings
                }
                all_metadata[annotated_filename] = metadata_entry
                logger.info(f"Successfully processed and gathered metadata for {img_path.name} -> {annotated_filename}")

            except Exception as e:
                logger.error(f"CRITICAL ERROR processing image {img_path.name}: {e}", exc_info=True)
                # Optionally add a placeholder entry for failed images
                # all_metadata[f"{prefix}_failed_{idx+1:04d}.error"] = {'original_path': str(img_path), 'error': str(e)}

        logger.info(f"Finished processing {len(all_metadata)} images for prefix '{prefix}'.")
        return all_metadata

    def create_sqlite_database(self, db_path: Path, user_metadata: Dict, public_metadata: Dict):
        """Creates or updates an SQLite database with the processed metadata."""
        logger.info(f"Updating SQLite database at: {db_path}")
        db_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        conn = None
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Create table if it doesn't exist - includes all fields
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL UNIQUE,       -- Annotated image filename (e.g., user_0001_annotated.jpg)
                original_path TEXT,                 -- Path to the original uploaded file
                processed_path TEXT,                -- Relative path to the processed/annotated file
                title TEXT,                         -- Auto-generated title (e.g., "Location - Date")
                location TEXT,
                date TEXT,
                type TEXT NOT NULL,                 -- 'user' or 'public'
                openai_keywords TEXT,               -- JSON string list from OpenAI
                openai_description TEXT,
                impact_weight REAL DEFAULT 1.0,
                resnet_embedding TEXT,              -- JSON string of list/vector (can be large)
                detected_objects TEXT               -- JSON string list of object names
            )
            ''')
            logger.debug("Database table 'memories' ensured to exist.")

            # --- Insert/Update Data ---
            def insert_metadata(metadata_dict: Dict, memory_type: str):
                logger.debug(f"Inserting/updating metadata for type '{memory_type}'...")
                inserted_count = 0
                updated_count = 0
                failed_count = 0
                for filename_key, data in tqdm(metadata_dict.items(), desc=f"Updating DB ({memory_type})", unit="entry"):
                    try:
                        # Use annotated filename as the unique key in the DB
                        title = f"{data.get('location', 'N/A')} - {data.get('date', 'N/A')}"

                        # Convert lists/embeddings to JSON strings for storage
                        keywords_json = json.dumps(data.get('openai_keywords', []))
                        embedding_json = json.dumps(data.get('resnet_embedding')) # Will be 'null' if None
                        objects_json = json.dumps(data.get('detected_objects', []))

                        # Use INSERT OR REPLACE to handle updates based on unique filename
                        cursor.execute(
                            '''
                            INSERT OR REPLACE INTO memories
                            (filename, original_path, processed_path, title, location, date, type, openai_keywords, openai_description, impact_weight, resnet_embedding, detected_objects)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ''',
                            (
                                filename_key, # The annotated filename (e.g., user_0001_annotated.jpg)
                                data.get('original_path'),
                                data.get('processed_path'),
                                title,
                                data.get('location', 'Unknown Location'),
                                data.get('date', 'Unknown Date'),
                                memory_type,
                                keywords_json,
                                data.get('openai_description', ''),
                                data.get('impact_weight', 1.0),
                                embedding_json,
                                objects_json
                            )
                        )
                        # Check if insert or replace occurred (optional, requires checking changes)
                        if cursor.rowcount > 0:
                             # This doesn't easily distinguish insert vs replace in standard sqlite3
                             # Assume success means inserted or replaced
                             inserted_count += 1 # Count successful operations
                        else:
                             # Should not happen with INSERT OR REPLACE unless error
                             logger.warning(f"No rows affected for {filename_key}, potentially unexpected.")
                             failed_count +=1


                    except sqlite3.Error as e:
                        logger.error(f"SQLite error inserting/replacing {filename_key} ({memory_type}): {e}")
                        failed_count += 1
                    except Exception as e:
                        logger.error(f"Unexpected error processing DB entry for {filename_key} ({memory_type}): {e}")
                        failed_count += 1

                logger.info(f"Finished DB update for '{memory_type}'. Success: {inserted_count}, Failed: {failed_count}.")

            # Insert user and public metadata
            insert_metadata(user_metadata, 'user')
            insert_metadata(public_metadata, 'public') # Will do nothing if public_metadata is empty

            conn.commit() # Commit changes
            logger.info(f"Database {db_path} updated successfully.")

        except sqlite3.Error as e:
            logger.error(f"Failed to connect to or operate on database {db_path}: {e}", exc_info=True)
        finally:
            if conn:
                conn.close() # Ensure connection is closed


# --- FastAPI Models ---
class ProcessRequest(BaseModel):
    session_id: str

class PhotoUploadResponse(BaseModel):
    session_id: str
    uploaded_files: List[str]
    total_files: int
    status: str

#=============
# Add a response model for the /process endpoint
class ProcessInitiatedResponse(BaseModel):
    session_id: str
    status: str
    message: str

class StatusResponse(BaseModel):
    session_id: str
    status: str
    message: str
    user_photos_processed: Optional[int] = None
    public_photos_processed: Optional[int] = None
    locations_detected: Optional[List[str]] = None
    
class ProcessedPhotoDetail(BaseModel):
    id: int
    filename: str
    original_path: Optional[str] = None
    processed_path: Optional[str] = None
    title: Optional[str] = None
    location: Optional[str] = None
    date: Optional[str] = None
    type: str  # 'user' or 'public'
    openai_keywords: List[str] = []
    openai_description: Optional[str] = None
    impact_weight: float = 1.0
    detected_objects: List[str] = []
    resnet_embedding: Optional[List[float]] = None  # Optional, can be large
    image_url: Optional[str] = None

class ProcessedPhotosResponse(BaseModel):
    session_id: str
    total_count: int
    user_count: int
    public_count: int
    photos: List[ProcessedPhotoDetail]

class ProcessedPhotoSummary(BaseModel):
    session_id: str
    total_photos: int
    user_photos: int
    public_photos: int
    locations_detected: List[str]
    date_range: Dict[str, Optional[str]]  # {"earliest": "2023-01-01", "latest": "2023-12-31"}
    top_keywords: List[Dict[str, Any]]  # [{"keyword": "beach", "count": 5}, ...]
    average_impact_weight: float

# Add this near the other Pydantic models at the top of the file
class FetchPublicRequest(BaseModel):
    session_id: str
    max_photos_per_location: int = Field(default=10, ge=1, le=20)
    total_limit: int = Field(default=100, ge=1, le=200)
    custom_locations: Optional[List[str]] = Field(default=None, description="Custom locations added by user")


# --- FastAPI Endpoints ---

@router.post("/photos", response_model=PhotoUploadResponse)
async def upload_photos(
    session_id: str = Query(..., description="Unique session identifier provided by the client."),
    files: List[UploadFile] = File(..., description="List of image files to upload.")
):
    """
    Uploads user photos to a temporary session directory for later processing.
    Handles multiple file uploads and basic validation.
    """
    logger.info(f"Received photo upload request for session_id: {session_id} with {len(files)} file(s).")
    session_manager = get_session_manager()
    if not session_manager:
         logger.error("Session Manager is not available.")
         raise HTTPException(status_code=500, detail="Server configuration error: Session Manager unavailable.")

    paths = session_manager.get_session_paths(session_id)
    if not paths or "raw_user" not in paths:
        logger.error(f"Session not found or 'raw_user' path missing for session_id: {session_id}")
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found or invalid.")

    raw_user_dir = Path(paths["raw_user"])
    saved_files_list = []
    file_count = 0

    try:
        os.makedirs(raw_user_dir, exist_ok=True) # Ensure directory exists
        logger.debug(f"Upload directory: {raw_user_dir}")

        for file in files:
            file_count += 1
            logger.debug(f"Processing uploaded file {file_count}/{len(files)}: {file.filename}")
            # Basic validation
            if not file.filename:
                 logger.warning(f"Skipping file {file_count} due to missing filename.")
                 continue
            if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.gif')):
                 logger.warning(f"Skipping file '{file.filename}' due to unsupported extension.")
                 continue

            # Sanitize filename (important for security)
            safe_filename = os.path.basename(file.filename)
            file_path = raw_user_dir / safe_filename

            try:
                # Use aiofiles for async write operation
                async with aiofiles.open(file_path, 'wb') as f:
                    content = await file.read() # Read file content async
                    await f.write(content)
                saved_files_list.append(safe_filename)
                logger.info(f"Successfully saved file: {file_path}")
            except Exception as e:
                logger.error(f"Error saving file '{safe_filename}' to {file_path}: {e}", exc_info=True)
            finally:
                 await file.close() # Ensure file handle is closed

    except Exception as e:
        logger.error(f"Critical error during file upload process for session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred during file upload.")

    status_msg = "success"
    if not saved_files_list:
        if file_count > 0:
            status_msg = "no_valid_files_saved"
            logger.warning(f"Upload completed for session {session_id}, but no valid files were saved.")
        else:
            status_msg = "no_files_uploaded"
            logger.warning(f"Upload request for session {session_id} contained no files.")


    logger.info(f"Upload complete for session {session_id}. Saved {len(saved_files_list)} files.")
    return PhotoUploadResponse(
        session_id=session_id,
        uploaded_files=saved_files_list,
        total_files=len(saved_files_list),
        status=status_msg
    )

# --- Background Task Function ---
# This function runs the actual processing. It's called by the /process endpoint.
# Note: This function itself runs synchronously within the background task runner.
async def process_session_photos_task(processor: DataProcessor, session_id: str, paths: dict):
    """
    Background task to process photos for a session using DataProcessor.
    Updates the database upon completion.
    """
    logger.info(f"[Background Task] Starting processing for session: {session_id}")
    try:
        # Step 1: Process user images using DataProcessor
        user_metadata = processor.process_images(
            source_dir=Path(paths["raw_user"]),
            output_dir=Path(paths["processed_user"]),
            prefix="user"
        )

        # Step 2: Define DB path and save metadata to the database
        db_path = Path(paths.get("metadata", ".")) / f"{session_id}_memories.db"
        paths["db_path"] = str(db_path) # Store for status check

        processor.create_sqlite_database(
            db_path=db_path,
            user_metadata=user_metadata,
            public_metadata={} # Assuming only user photos processed here
        )
        logger.info(f"[Background Task] Completed processing and DB update for session: {session_id} at {db_path}")

        # Step 3 (Optional): Update session manager with locations
        session_manager = get_session_manager()
        if session_manager:
            try:
                locations_found = set()
                for data in user_metadata.values():
                    location = data.get('location')
                    if location and location != "Unknown Location":
                        locations_found.add(location.split(',')[0].strip())
                logger.debug(f"[Background Task] Adding locations to session manager: {locations_found}")
                for loc in locations_found:
                    session_manager.add_location(session_id, loc)
            except Exception as sm_e:
                logger.error(f"[Background Task] Failed to update session manager locations for {session_id}: {sm_e}")

    except Exception as e:
        # Log errors that occur within the background task
        logger.error(f"[Background Task] CRITICAL error during processing for session {session_id}: {e}", exc_info=True)
        # Note: We can't directly return an HTTP response from a background task.
        # The status endpoint should reflect the lack of completion or an error state if possible.

# --- Modified /process endpoint ---
@router.post("/process", response_model=ProcessInitiatedResponse)
async def process_photos(
    request: ProcessRequest, # Takes session_id from JSON body
    background_tasks: BackgroundTasks # Inject BackgroundTasks dependency
):
    """
    Initiates the processing of uploaded photos for the given session
    by adding the task to the background. Returns immediately.
    """
    session_id = request.session_id
    logger.info(f"Received request to initiate background processing for session_id: {session_id}")

    session_manager = get_session_manager()
    if not session_manager:
         logger.error("Session Manager is not available for /process endpoint.")
         raise HTTPException(status_code=500, detail="Server configuration error: Session Manager unavailable.")

    paths = session_manager.get_session_paths(session_id)
    if not paths or not all(k in paths for k in ["raw_user", "processed_user", "metadata", "raw_public", "processed_public"]):
        logger.error(f"Session '{session_id}' not found or missing required paths for processing.")
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found or configuration is invalid.")

    # Ensure output directories exist before potentially starting the task
    try:
        os.makedirs(paths.get("processed_user"), exist_ok=True)
        os.makedirs(paths.get("processed_public"), exist_ok=True)
        os.makedirs(paths.get("metadata"), exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create necessary output directories for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to prepare session directories for processing.")

    try:
        # Initialize the DataProcessor - This still happens synchronously here
        # to catch init errors before starting the background task.
        logger.debug(f"Initializing DataProcessor for session {session_id} before background task...")
        processor = DataProcessor(
            raw_user_dir=paths["raw_user"],
            raw_public_dir=paths["raw_public"],
            processed_user_dir=paths["processed_user"],
            processed_public_dir=paths["processed_public"],
            metadata_dir=paths["metadata"],
        )
        logger.debug("DataProcessor initialized successfully.")

        # Add the actual processing function to run in the background
        background_tasks.add_task(process_session_photos_task, processor, session_id, paths)
        logger.info(f"Added processing task for session {session_id} to background.")

        # Return immediately, indicating processing has started
        return ProcessInitiatedResponse(
            session_id=session_id,
            status="processing_started",
            message="Photo processing started in the background. Use the /status endpoint to check progress.",
            # openai_keywords=processor.openai_keywords,
            # openai_description=processor.openai_des,
            # location=processor.location,
            # impact_weight=processor.impact_weight,
            # restnet_embedding=processor.restnet_embedding
        )

    except Exception as e:
        # Catch errors during DataProcessor init or other setup before background task starts
        logger.error(f"Error initiating processing for session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to initiate photo processing: {str(e)}")


# --- Status Endpoint ---
# (Keep the /status/{session_id} endpoint as defined previously - it reads the DB)
@router.get("/status/{session_id}", response_model=StatusResponse)
async def get_processing_status(session_id: str):
    """
    Checks the processing status by querying the database created by the background task.
    """
    logger.info(f"Received status check request for session_id: {session_id}")
    session_manager = get_session_manager()
    if not session_manager:
         logger.error("Session Manager is not available for /status endpoint.")
         raise HTTPException(status_code=500, detail="Server configuration error: Session Manager unavailable.")

    paths = session_manager.get_session_paths(session_id)
    if not paths or "metadata" not in paths:
        logger.warning(f"Session '{session_id}' not found or metadata path missing for status check.")
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found or configuration is invalid.")

    # Define the expected database path
    db_path = Path(paths.get("metadata")) / f"{session_id}_memories.db"
    logger.debug(f"Checking for database at: {db_path}")

    # Check if processing might still be ongoing (no DB yet)
    if not db_path.exists():
        # Check if raw files exist to distinguish "not started" vs "in progress"
        raw_user_dir = Path(paths.get("raw_user", ""))
        status_msg = "pending"
        message = "Processing is pending or has not created the database yet."
        if raw_user_dir.exists() and any(raw_user_dir.iterdir()):
             status_msg = "processing_pending_db"
             message = "Session files exist, processing is likely ongoing or failed before DB creation."
        else:
             status_msg = "no_files_or_not_started"
             message = "No files found in session, or processing not initiated."

        # Return pending status
        return StatusResponse(session_id=session_id, status=status_msg, message=message)

    # Database exists, query it for status
    conn = None
    try:
        logger.debug(f"Connecting to database {db_path} for status check.")
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Get counts per type
        cursor.execute("SELECT type, COUNT(*) FROM memories GROUP BY type")
        counts = {row[0]: row[1] for row in cursor.fetchall()}
        user_count = counts.get("user", 0)
        public_count = counts.get("public", 0) # Check even if not expected in this flow
        logger.debug(f"DB counts - User: {user_count}, Public: {public_count}")

        # Get distinct primary locations for user photos
        cursor.execute("SELECT DISTINCT location FROM memories WHERE type='user'")
        locations = list(set(
             row[0].split(',')[0].strip() for row in cursor.fetchall()
             if row[0] and row[0].lower() != "unknown location" # Filter out unknowns
        ))
        logger.debug(f"DB distinct locations: {locations}")

        # Determine status based on DB content (adjust logic as needed)
        # This needs refinement - how do we know if the background task *finished* vs just created some entries?
        # We might need a separate status flag in the DB or session manager updated by the task.
        # For now, presence of user photos indicates user processing is done.
        status_val = "unknown_db_state"
        message = "Database exists, but processing state is unclear."
        if user_count > 0:
            # Assuming user photo processing is the main task tracked here
            status_val = "user_processed" # Matches frontend expectation
            message = f"User photo processing complete. Found {user_count} entries."
        else:
             status_val = "db_exists_no_user_photos"
             message = "Database exists, but no 'user' type entries were found."
             # If public photos are processed separately, add logic here

        # TODO: Add a more robust way to track if the background task itself has fully completed or errored.
        # This might involve writing a status file or updating the session manager state upon task completion/failure.

        return StatusResponse(
            session_id=session_id,
            status=status_val, # Return status based on DB content
            message=message,
            user_photos_processed=user_count,
            public_photos_processed=public_count, # If applicable
            locations_detected=locations
        )

    except sqlite3.Error as e:
        logger.error(f"SQLite error checking status for {session_id} in {db_path}: {e}", exc_info=True)
        # Return an error status that the frontend can understand
        return StatusResponse(
            session_id=session_id,
            status="error_db_read",
            message=f"Error reading database: {str(e)}"
        )
    except Exception as e:
         logger.error(f"Unexpected error during status check for {session_id}: {e}", exc_info=True)
         raise HTTPException(status_code=500, detail=f"Unexpected error checking status: {str(e)}")
    finally:
        if conn:
            conn.close()
            logger.debug("Database connection closed.")

# ================
@router.get("/processed/{session_id}", response_model=ProcessedPhotosResponse)
async def get_processed_photos(
    session_id: str,
    photo_type: str = Query('all', description="Filter by type: 'all', 'user', or 'public'"),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of photos to return"),
    offset: int = Query(0, ge=0, description="Number of photos to skip"),
    include_embeddings: bool = Query(True, description="Include ResNet embeddings in response")
):
    """
    Get details of all processed photos for a session.
    """
    logger.info(f"Received request for processed photos: session_id={session_id}, type={photo_type}, limit={limit}")
    
    session_manager = get_session_manager()
    if not session_manager:
        raise HTTPException(status_code=500, detail="Session Manager unavailable.")
    
    paths = session_manager.get_session_paths(session_id)
    if not paths or "metadata" not in paths:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    
    # Check database exists
    db_path = Path(paths["metadata"]) / f"{session_id}_memories.db"
    if not db_path.exists():
        raise HTTPException(status_code=404, detail=f"No processed photos found for session '{session_id}'.")
    
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Build query based on photo_type filter
        base_query = """
            SELECT id, filename, original_path, processed_path, title, location, date, type,
                   openai_keywords, openai_description, impact_weight, detected_objects
        """
        if include_embeddings:
            base_query += ", resnet_embedding"
        
        base_query += " FROM memories"
        
        params = []
        if photo_type != 'all':
            base_query += " WHERE type = ?"
            params.append(photo_type)
        
        # Add ordering and pagination
        base_query += " ORDER BY impact_weight DESC, date DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        cursor.execute(base_query, params)
        rows = cursor.fetchall()
        
        # Get total counts
        cursor.execute("SELECT type, COUNT(*) FROM memories GROUP BY type")
        counts = {row[0]: row[1] for row in cursor.fetchall()}
        total_count = sum(counts.values())
        user_count = counts.get('user', 0)
        public_count = counts.get('public', 0)
        
        # Process results
        photos = []
        base_static_path = f"/api/session/static/{session_id}"
        
        for row in rows:
            # Parse JSON fields
            keywords = json.loads(row['openai_keywords']) if row['openai_keywords'] else []
            objects = json.loads(row['detected_objects']) if row['detected_objects'] else []
            
            # Handle embedding if requested
            embedding = None
            if include_embeddings and 'resnet_embedding' in row.keys() and row['resnet_embedding']:
                try:
                    embedding = json.loads(row['resnet_embedding'])
                except json.JSONDecodeError:
                    embedding = None
            
            # Construct image URL
            img_url = None
            if row['processed_path']:
                # Extract type and filename from processed_path
                path_parts = Path(row['processed_path']).parts
                if len(path_parts) >= 2:
                    img_type = path_parts[-2]  # 'user' or 'public'
                    img_filename = path_parts[-1]
                    img_url = f"{base_static_path}/{img_type}/{img_filename}"
            
            photo_detail = ProcessedPhotoDetail(
                id=row['id'],
                filename=row['filename'],
                original_path=row['original_path'],
                processed_path=row['processed_path'],
                title=row['title'],
                location=row['location'],
                date=row['date'],
                type=row['type'],
                openai_keywords=keywords,
                openai_description=row['openai_description'],
                impact_weight=row['impact_weight'],
                detected_objects=objects,
                resnet_embedding=embedding,
                image_url=img_url
            )
            photos.append(photo_detail)
        
        return ProcessedPhotosResponse(
            session_id=session_id,
            total_count=total_count,
            user_count=user_count,
            public_count=public_count,
            photos=photos
        )
        
    except sqlite3.Error as e:
        logger.error(f"Database error getting processed photos for {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        logger.error(f"Error getting processed photos for {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    finally:
        if conn:
            conn.close()


@router.get("/processed/{session_id}/summary", response_model=ProcessedPhotoSummary)
async def get_processed_photos_summary(session_id: str):
    """
    Get a summary of processed photos for a session including statistics.
    """
    logger.info(f"Received request for processed photos summary: session_id={session_id}")
    
    session_manager = get_session_manager()
    if not session_manager:
        raise HTTPException(status_code=500, detail="Session Manager unavailable.")
    
    paths = session_manager.get_session_paths(session_id)
    if not paths or "metadata" not in paths:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    
    # Check database exists
    db_path = Path(paths["metadata"]) / f"{session_id}_memories.db"
    if not db_path.exists():
        raise HTTPException(status_code=404, detail=f"No processed photos found for session '{session_id}'.")
    
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get basic counts
        cursor.execute("SELECT type, COUNT(*) FROM memories GROUP BY type")
        counts = {row[0]: row[1] for row in cursor.fetchall()}
        total_photos = sum(counts.values())
        user_photos = counts.get('user', 0)
        public_photos = counts.get('public', 0)
        
        # Get unique locations
        cursor.execute("""
            SELECT DISTINCT location FROM memories 
            WHERE location IS NOT NULL AND location != 'Unknown Location'
        """)
        locations = [row[0].split(',')[0].strip() for row in cursor.fetchall()]
        locations = list(set(locations))  # Remove duplicates
        
        # Get date range
        cursor.execute("SELECT MIN(date) as earliest, MAX(date) as latest FROM memories")
        date_row = cursor.fetchone()
        date_range = {
            "earliest": date_row['earliest'],
            "latest": date_row['latest']
        }
        
        # Get average impact weight
        cursor.execute("SELECT AVG(impact_weight) as avg_weight FROM memories")
        avg_weight = cursor.fetchone()['avg_weight'] or 1.0
        
        # Get top keywords
        cursor.execute("SELECT openai_keywords FROM memories")
        all_keywords = []
        for row in cursor.fetchall():
            if row['openai_keywords']:
                try:
                    keywords = json.loads(row['openai_keywords'])
                    all_keywords.extend(keywords)
                except json.JSONDecodeError:
                    continue
        
        # Count keyword frequency
        keyword_counts = {}
        for keyword in all_keywords:
            keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
        
        # Get top 10 keywords
        top_keywords = [
            {"keyword": k, "count": v} 
            for k, v in sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        ]
        
        return ProcessedPhotoSummary(
            session_id=session_id,
            total_photos=total_photos,
            user_photos=user_photos,
            public_photos=public_photos,
            locations_detected=locations,
            date_range=date_range,
            top_keywords=top_keywords,
            average_impact_weight=round(avg_weight, 2)
        )
        
    except sqlite3.Error as e:
        logger.error(f"Database error getting summary for {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        logger.error(f"Error getting summary for {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    finally:
        if conn:
            conn.close()


@router.get("/processed/{session_id}/photo/{photo_id}", response_model=ProcessedPhotoDetail)
async def get_single_processed_photo(session_id: str, photo_id: int, include_embeddings: bool = Query(True)):
    """
    Get details of a single processed photo by ID.
    """
    logger.info(f"Received request for single photo: session_id={session_id}, photo_id={photo_id}")
    
    session_manager = get_session_manager()
    if not session_manager:
        raise HTTPException(status_code=500, detail="Session Manager unavailable.")
    
    paths = session_manager.get_session_paths(session_id)
    if not paths or "metadata" not in paths:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    
    # Check database exists
    db_path = Path(paths["metadata"]) / f"{session_id}_memories.db"
    if not db_path.exists():
        raise HTTPException(status_code=404, detail=f"No processed photos found for session '{session_id}'.")
    
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Build query
        query = """
            SELECT id, filename, original_path, processed_path, title, location, date, type,
                   openai_keywords, openai_description, impact_weight, detected_objects
        """
        if include_embeddings:
            query += ", resnet_embedding"
        
        query += " FROM memories WHERE id = ?"
        
        cursor.execute(query, (photo_id,))
        row = cursor.fetchone()
        
        if not row:
            raise HTTPException(status_code=404, detail=f"Photo with ID {photo_id} not found.")
        
        # Parse JSON fields
        keywords = json.loads(row['openai_keywords']) if row['openai_keywords'] else []
        objects = json.loads(row['detected_objects']) if row['detected_objects'] else []
        
        # Handle embedding if requested
        embedding = None
        if include_embeddings and 'resnet_embedding' in row.keys() and row['resnet_embedding']:
            try:
                embedding = json.loads(row['resnet_embedding'])
            except json.JSONDecodeError:
                embedding = None
        
        # Construct image URL
        img_url = None
        base_static_path = f"/api/session/static/{session_id}"
        if row['processed_path']:
            # Extract type and filename from processed_path
            path_parts = Path(row['processed_path']).parts
            if len(path_parts) >= 2:
                img_type = path_parts[-2]  # 'user' or 'public'
                img_filename = path_parts[-1]
                img_url = f"{base_static_path}/{img_type}/{img_filename}"
        
        return ProcessedPhotoDetail(
            id=row['id'],
            filename=row['filename'],
            original_path=row['original_path'],
            processed_path=row['processed_path'],
            title=row['title'],
            location=row['location'],
            date=row['date'],
            type=row['type'],
            openai_keywords=keywords,
            openai_description=row['openai_description'],
            impact_weight=row['impact_weight'],
            detected_objects=objects,
            resnet_embedding=embedding,
            image_url=img_url
        )
        
    except sqlite3.Error as e:
        logger.error(f"Database error getting photo {photo_id} for {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        logger.error(f"Error getting photo {photo_id} for {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    finally:
        if conn:
            conn.close()

@router.post("/fetch-public")
async def fetch_public_photos(request: FetchPublicRequest):
    """
    Fetch public photos from Unsplash based on detected locations in the session
    and/or custom locations provided by the user.
    """
    session_id = request.session_id
    max_photos_per_location = request.max_photos_per_location
    total_limit = request.total_limit
    custom_locations = request.custom_locations
    
    logger.info(f"Fetching public photos for session: {session_id}")
    logger.info(f"Custom locations provided: {custom_locations}")
    
    session_manager = get_session_manager()
    if not session_manager:
        raise HTTPException(status_code=500, detail="Session Manager unavailable.")
    
    paths = session_manager.get_session_paths(session_id)
    if not paths or "metadata" not in paths:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    
    # Get detected locations from the database
    db_path = Path(paths["metadata"]) / f"{session_id}_memories.db"
    detected_locations = []
    
    if db_path.exists():
        conn = None
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Get unique locations from user memories
            cursor.execute("""
                SELECT DISTINCT location FROM memories 
                WHERE type = 'user' AND location != 'Unknown Location' AND location IS NOT NULL
            """)
            
            rows = cursor.fetchall()
            detected_locations = [row[0].split(',')[0].strip() for row in rows]  # Get primary location part
            detected_locations = list(set(detected_locations))  # Remove duplicates
            
            logger.info(f"Found {len(detected_locations)} detected locations: {detected_locations}")
            
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            # Continue with empty detected_locations
        finally:
            if conn:
                conn.close()
    
    # Combine detected locations with custom locations
    all_locations = detected_locations.copy()
    
    if custom_locations:
        # Clean and validate custom locations
        cleaned_custom = []
        for loc in custom_locations:
            if loc and isinstance(loc, str) and loc.strip():
                clean_loc = loc.strip()
                if clean_loc not in all_locations:  # Avoid duplicates
                    cleaned_custom.append(clean_loc)
                    all_locations.append(clean_loc)
        
        logger.info(f"Added {len(cleaned_custom)} custom locations: {cleaned_custom}")
    
    # Remove duplicates and filter out empty strings
    all_locations = list(set([loc for loc in all_locations if loc]))
    
    if not all_locations:
        return {
            "status": "no_locations",
            "message": "No locations found to fetch public photos. Please add locations manually.",
            "detected_locations": detected_locations,
            "custom_locations": custom_locations or [],
            "total_locations": 0,
            "photos_fetched": 0
        }
    
    logger.info(f"Total locations to process: {len(all_locations)} - {all_locations}")
    
    # Initialize the public photo fetcher
    try:
        fetcher = PublicPhotoFetcher()  # Use the enhanced fetcher
        
        # Fetch photos for each location
        all_photos = []
        photos_per_location = min(max_photos_per_location, total_limit // len(all_locations))
        
        for location in all_locations:
            try:
                logger.info(f"Fetching photos for location: {location}")
                photos = await fetcher.fetch_photos_for_location(
                    location, 
                    limit=photos_per_location
                )
                
                # Tag photos with whether they came from detected or custom locations
                for photo in photos:
                    photo['source_type'] = 'detected' if location in detected_locations else 'custom'
                    photo['source_location'] = location
                
                all_photos.extend(photos)
                
                # Check total limit
                if len(all_photos) >= total_limit:
                    all_photos = all_photos[:total_limit]
                    break
                    
            except Exception as e:
                logger.error(f"Error fetching photos for {location}: {e}")
                continue
        
        # Save photos to the session
        if all_photos:
            await save_public_photos_to_session(session_id, paths, all_photos)
        
        return {
            "status": "public_fetch_started" if all_photos else "no_photos_found",
            "message": f"Successfully fetched {len(all_photos)} public photos from {len(all_locations)} locations.",
            "detected_locations": detected_locations,
            "custom_locations": custom_locations or [],
            "total_locations": len(all_locations),
            "photos_fetched": len(all_photos),
            "photos_per_location": photos_per_location
        }
        
    except Exception as e:
        logger.error(f"Error in public photo fetching: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error fetching public photos: {str(e)}")

class PublicPhotoFetcher:
    """Fetches public photos from various sources based on location."""
    
    def __init__(self):
        # Get your Unsplash API key from environment variables
        # Sign up at https://unsplash.com/developers to get a free API key
        self.unsplash_api_key = os.getenv("UNSPLASH_ACCESS_KEY")
        self.unsplash_base_url = "https://api.unsplash.com"
        
        if not self.unsplash_api_key:
            logger.warning("UNSPLASH_ACCESS_KEY not found. Public photo fetching will be limited.")
    
    async def fetch_photos_for_location(self, location: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Fetch photos for a specific location from Unsplash."""
        photos = []
        
        if self.unsplash_api_key:
            photos = await self._fetch_from_unsplash(location, limit)
        
        # Fallback to placeholder images if no API key or no results
        if not photos:
            photos = self._generate_placeholder_photos(location, limit)
        
        return photos
    
    async def _fetch_from_unsplash(self, location: str, limit: int) -> List[Dict[str, Any]]:
        """Fetch photos from Unsplash API."""
        photos = []
        
        try:
            headers = {
                "Authorization": f"Client-ID {self.unsplash_api_key}",
                "Accept-Version": "v1"
            }
            
            # Search for photos with location-based queries
            search_queries = [
                f"{location} city",
                f"{location} architecture",
                f"{location} landmarks",
                f"{location} street",
                location
            ]
            
            async with aiohttp.ClientSession() as session:
                for query in search_queries:
                    if len(photos) >= limit:
                        break
                    
                    url = f"{self.unsplash_base_url}/search/photos"
                    params = {
                        "query": query,
                        "per_page": min(10, limit - len(photos)),
                        "orientation": "landscape",
                        "order_by": "relevant"
                    }
                    
                    async with session.get(url, headers=headers, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            for photo in data.get("results", []):
                                if len(photos) >= limit:
                                    break
                                
                                # Extract photo information
                                photo_info = {
                                    "url": photo["urls"]["regular"],
                                    "thumbnail_url": photo["urls"]["small"],
                                    "title": photo.get("description") or photo.get("alt_description") or f"{location} - Public Photo",
                                    "location": location,
                                    "attribution": {
                                        "photographer": photo["user"]["name"],
                                        "photographer_url": photo["user"]["links"]["html"],
                                        "unsplash_url": photo["links"]["html"],
                                        "unsplash_id": photo["id"]
                                    },
                                    "width": photo["width"],
                                    "height": photo["height"],
                                    "search_query": query
                                }
                                photos.append(photo_info)
                        else:
                            logger.warning(f"Unsplash API error for query '{query}': {response.status}")
                
                logger.info(f"Fetched {len(photos)} photos from Unsplash for {location}")
                
        except Exception as e:
            logger.error(f"Error fetching from Unsplash for {location}: {e}")
        
        return photos
    
    def _generate_placeholder_photos(self, location: str, limit: int) -> List[Dict[str, Any]]:
        """Generate placeholder photos when real photos are not available."""
        photos = []
        
        # Use placeholder image services
        placeholder_services = [
            "https://picsum.photos/800/600",
            "https://source.unsplash.com/800x600",
            "https://images.unsplash.com/photo-1500530855697-b586d89ba3ee"
        ]
        
        for i in range(min(limit, 3)):  # Limit placeholder photos
            service_url = placeholder_services[i % len(placeholder_services)]
            
            photo_info = {
                "url": f"{service_url}?sig={i}",
                "thumbnail_url": f"{service_url}?sig={i}",
                "title": f"{location} - Placeholder Photo {i+1}",
                "location": location,
                "attribution": {
                    "photographer": "Placeholder Service",
                    "photographer_url": "#",
                    "source": "Placeholder",
                    "note": "This is a placeholder image"
                },
                "width": 800,
                "height": 600,
                "search_query": location,
                "is_placeholder": True
            }
            photos.append(photo_info)
        
        logger.info(f"Generated {len(photos)} placeholder photos for {location}")
        return photos


# Add this function to save photos to the session

async def save_public_photos_to_session(session_id: str, paths: Dict[str, str], photos: List[Dict[str, Any]]):
    """Save public photos to the session database and download images."""
    logger.info(f"Saving {len(photos)} public photos to session {session_id}")
    
    # Prepare directories
    public_raw_dir = Path(paths["raw_public"])
    public_processed_dir = Path(paths["processed_public"])
    os.makedirs(public_raw_dir, exist_ok=True)
    os.makedirs(public_processed_dir, exist_ok=True)
    
    # Database connection
    db_path = Path(paths["metadata"]) / f"{session_id}_memories.db"
    conn = None
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Download and save photos
        async with aiohttp.ClientSession() as session:
            for i, photo in enumerate(photos):
                try:
                    # Generate filename
                    filename = f"public_{i+1:04d}.jpg"
                    file_path = public_processed_dir / filename
                    
                    # Download image if not placeholder
                    if not photo.get("is_placeholder", False):
                        async with session.get(photo["url"]) as response:
                            if response.status == 200:
                                with open(file_path, 'wb') as f:
                                    f.write(await response.read())
                            else:
                                logger.warning(f"Failed to download {photo['url']}")
                                continue
                    else:
                        # For placeholders, just create a small info file
                        with open(file_path.with_suffix('.txt'), 'w') as f:
                            f.write(f"Placeholder for {photo['location']}")
                    
                    # Create database entry with source information
                    title = f"{photo['location']} - {photo['title']}"
                    openai_keywords = [photo['location'], 'public', 'landmark', 'city']
                    
                    # Add source type to keywords
                    source_type = photo.get('source_type', 'detected')
                    openai_keywords.append(f"{source_type}_location")
                    
                    # Include attribution and source info in description
                    attribution = photo.get('attribution', {})
                    description_parts = [photo.get('title', '')]
                    if attribution.get('photographer'):
                        description_parts.append(f"Photo by {attribution['photographer']}")
                    if source_type == 'custom':
                        description_parts.append(f"From manually added location: {photo.get('source_location', photo['location'])}")
                    
                    attribution_text = " | ".join(filter(None, description_parts))
                    
                    cursor.execute('''
                        INSERT INTO memories 
                        (filename, original_path, processed_path, title, location, date, type, 
                         openai_keywords, openai_description, impact_weight, detected_objects)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        filename,
                        str(file_path),  # original_path
                        f"processed/public/{filename}",  # processed_path
                        title,
                        photo['location'],
                        datetime.now().strftime('%Y-%m-%d'),
                        'public',
                        json.dumps(openai_keywords),
                        attribution_text,
                        1.0,  # Default weight
                        json.dumps(['landmark', 'public_photo', source_type])
                    ))
                    
                except Exception as e:
                    logger.error(f"Error saving public photo {i}: {e}")
                    continue
        
        conn.commit()
        logger.info(f"Successfully saved {len(photos)} public photos to database")
        
    except Exception as e:
        logger.error(f"Error saving public photos: {e}", exc_info=True)
    finally:
        if conn:
            conn.close()

# --- Include router in main app ---
# Example main.py:
# from fastapi import FastAPI
# from app.api.endpoints import upload # Assuming this file is app/api/endpoints/upload.py
#
# app = FastAPI()
# app.include_router(upload.router, prefix="/api/upload", tags=["upload"])
