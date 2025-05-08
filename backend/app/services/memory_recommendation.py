# memory_recommendation.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
from datetime import datetime
from PIL import Image
import cv2
import torch
import torchvision.transforms as T
from torchvision.models import resnet50
# from ultralytics import YOLO
import os
import base64
from io import BytesIO

# Import YOLO for object detection
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("YOLO not available. Install with 'pip install ultralytics'")
    YOLO_AVAILABLE = False

class MemoryRecommendationEngine:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        
        # Initialize ResNet model for visual feature extraction
        self.image_model = self._initialize_image_model()
        
        # Initialize YOLO model for object detection
        self.yolo_model = self._initialize_yolo_model()
        
        # Standard image transformations for ResNet input
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def _initialize_image_model(self):
        """Initialize the pre-trained ResNet model for feature extraction."""
        try:
            model = resnet50(pretrained=True)
            model.eval()  # Set to evaluation mode
            return model
        except Exception as e:
            print(f"Error initializing ResNet model: {e}")
            return None
    
    def _initialize_yolo_model(self):
        """Initialize the pre-trained YOLO model for object detection."""
        if not YOLO_AVAILABLE:
            return None
            
        try:
            # Load a pre-trained YOLO model (nano version for speed)
            model = YOLO('yolov8n.pt')
            return model
        except Exception as e:
            print(f"Error initializing YOLO model: {e}")
            return None
    
    def extract_image_features(self, image_path):
        """Extract feature vector from an image using ResNet50."""
        try:
            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}")
                return None
                
            # Load and preprocess the image
            image = Image.open(image_path).convert("RGB")
            img_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
            
            # Extract features
            with torch.no_grad():
                features = self.image_model(img_tensor).squeeze(0)
            
            return features.numpy()
        except Exception as e:
            print(f"Error extracting image features: {e}")
            return None
    
    def detect_objects_in_image(self, image_path, conf_threshold=0.25):
        """
        Detect objects in an image using YOLO and return both detections and visualized image.
        """
        print(f"Debug: Using confidence threshold: {conf_threshold}")
        
        if not YOLO_AVAILABLE:
            print("Debug: YOLO is not available")
            return None, None, []
        
        try:
            if not os.path.exists(image_path):
                print(f"Debug: Image not found: {image_path}")
                return None, None, []
                
            print(f"Debug: Image exists at {image_path}")
            
            if not self.yolo_model:
                print("Debug: YOLO model not initialized, trying to initialize now")
                self.yolo_model = self._initialize_yolo_model()
                if not self.yolo_model:
                    print("Debug: Failed to initialize YOLO model")
                    return None, None, []
            
            # Try loading the image with PIL to verify it's valid
            try:
                from PIL import Image
                test_img = Image.open(image_path)
                print(f"Debug: Successfully opened image with PIL, size: {test_img.size}")
                test_img.close()
            except Exception as e:
                print(f"Debug: Error opening image with PIL: {e}")
                return None, None, []
                
            # Run YOLO detection
            print(f"Debug: Running YOLO detection on {image_path}")
            results = self.yolo_model(image_path, conf=conf_threshold)
            print(f"Debug: YOLO detection complete. Results type: {type(results)}")
            
            if not results:
                print("Debug: No results returned from YOLO")
                return None, None, []
                
            # Check what's in results
            print(f"Debug: Number of detection results: {len(results)}")
            
            # Process results to get the image with bounding boxes
            result = results[0]
            print(f"Debug: Processing first result")
            
            # Get detected object classes and counts
            detected_classes = {}
            
            if hasattr(result, 'boxes') and hasattr(result.boxes, 'cls') and len(result.boxes.cls) > 0:
                print(f"Debug: Found {len(result.boxes.cls)} detection boxes")
                class_indices = result.boxes.cls.cpu().numpy()
                
                # Count occurrences of each class
                for idx in class_indices:
                    class_name = result.names[int(idx)]
                    detected_classes[class_name] = detected_classes.get(class_name, 0) + 1
                print(f"Debug: Detected classes: {detected_classes}")
            else:
                print("Debug: No detection boxes found or invalid format")
                
            # Get the processed image with bounding boxes
            try:
                annotated_img = result.plot()
                print("Debug: Successfully plotted annotated image")
                
                # Convert the image to base64 for easy display
                import cv2
                import base64
                _, buffer = cv2.imencode('.jpg', annotated_img)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                print("Debug: Successfully encoded image to base64")
            except Exception as e:
                print(f"Debug: Error creating annotated image: {e}")
                img_base64 = None
            
            # Format the detected objects list
            detected_objects = [
                {"label": class_name, "count": count, "confidence": 0.0}
                for class_name, count in detected_classes.items()
            ]
            
            return results, img_base64, detected_objects
                
        except Exception as e:
            print(f"Debug: Error in detect_objects_in_image: {e}")
            import traceback
            traceback.print_exc()
            return None, None, []
    
    def extract_image_with_objects(self, image_path, output_path=None):
        """
        Extract features and detect objects in an image, optionally saving the annotated image.
        
        Returns a dictionary with features, detected objects, and base64 encoded annotated image.
        """
        results = {
            "features": None,
            "objects_detected": [],
            "annotated_image_base64": None,
            "success": False
        }
        
        # Check if image exists
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return results
            
        # 1. Extract ResNet features for similarity comparison
        if self.image_model:
            results["features"] = self.extract_image_features(image_path)
            
        # 2. Detect objects with YOLO
        if self.yolo_model:
            yolo_results, annotated_image_base64, detected_objects = self.detect_objects_in_image(image_path)
            
            if detected_objects:
                results["objects_detected"] = detected_objects
                results["annotated_image_base64"] = annotated_image_base64
                
                # If output path is provided, save the annotated image
                if output_path and annotated_image_base64:
                    try:
                        # Convert base64 back to image and save
                        img_data = base64.b64decode(annotated_image_base64)
                        with open(output_path, 'wb') as f:
                            f.write(img_data)
                    except Exception as e:
                        print(f"Error saving annotated image: {e}")
        
        # Set success flag if either features or objects were detected
        results["success"] = results["features"] is not None or len(results["objects_detected"]) > 0
        
        return results
    
    def find_similar_images(self, query_image_path, dataset_paths, top_n=5):
        """Find similar images in a dataset based on feature similarity."""
        # Extract features from query image
        query_features = self.extract_image_features(query_image_path)
        if query_features is None:
            return []
        
        # Extract features from dataset images
        dataset_features = []
        valid_paths = []
        
        for path in dataset_paths:
            features = self.extract_image_features(path)
            if features is not None:
                dataset_features.append(features)
                valid_paths.append(path)
        
        if not dataset_features:
            return []
        
        # Calculate similarities
        dataset_features_array = np.vstack(dataset_features)
        similarities = cosine_similarity(query_features.reshape(1, -1), dataset_features_array)[0]
        
        # Sort by similarity and get top matches
        top_indices = similarities.argsort()[::-1][:top_n]
        
        # Return results with similarity scores
        results = [(valid_paths[i], similarities[i]) for i in top_indices]
        return results
    
    def find_similar_memories(self, source_memory, all_memories, top_n=5):
        """Find similar memories using TF-IDF and cosine similarity."""
        if not all_memories:
            return []
            
        # Prepare text for vectorization
        texts = [
            f"{m.get('title', '')} {m.get('description', '')} {' '.join(m.get('keywords', []))}" 
            for m in all_memories
        ]
        
        # Vectorize texts
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        
        # Vectorize source memory
        source_text = (
            f"{source_memory.get('title', '')} "
            f"{source_memory.get('description', '')} "
            f"{' '.join(source_memory.get('keywords', []))}"
        )
        source_vector = self.tfidf_vectorizer.transform([source_text])
        
        # Calculate cosine similarities
        similarities = cosine_similarity(source_vector, tfidf_matrix)[0]
        
        # Get top similar memories (excluding the source memory itself if it's in all_memories)
        sorted_indices = similarities.argsort()[::-1]
        
        # Filter out the source memory if it's in the list
        similar_indices = [
            idx for idx in sorted_indices 
            if all_memories[idx].get('id') != source_memory.get('id')
        ][:top_n]
        
        return [all_memories[idx] for idx in similar_indices]
    
    def find_memories_by_visual_similarity(self, query_image_path, memories_with_images, top_n=5):
        """
        Find memories with similar visual content based on feature similarity.
        
        Args:
            query_image_path: Path to the query image
            memories_with_images: List of memory dictionaries with 'filename' field
            top_n: Number of top matches to return
            
        Returns:
            List of matching memories with similarity scores added
        """
        if not memories_with_images:
            return []
            
        # Extract features from query image
        query_features = self.extract_image_features(query_image_path)
        if query_features is None:
            return []
            
        # Prepare memory images and features
        memory_features = []
        valid_memories = []
        
        for memory in memories_with_images:
            # Skip memories without image files
            if not memory.get('filename'):
                continue
                
            # Construct full image path
            memory_type = memory.get('type', 'user')
            if memory_type == 'user':
                image_dir = 'data/processed/user_photos/'
            else:
                image_dir = 'data/processed/public_photos/'
                
            image_path = os.path.join(image_dir, memory.get('filename'))
            
            # Get features
            features = self.extract_image_features(image_path)
            if features is not None:
                memory_features.append(features)
                valid_memories.append(memory)
        
        if not valid_memories:
            return []
            
        # Calculate similarities
        memory_features_array = np.vstack(memory_features)
        similarities = cosine_similarity(query_features.reshape(1, -1), memory_features_array)[0]
        
        # Sort memories by similarity
        for i, similarity in enumerate(similarities):
            valid_memories[i]['similarity'] = float(similarity)
            
        sorted_memories = sorted(valid_memories, key=lambda m: m.get('similarity', 0), reverse=True)
        
        # Return top matches
        return sorted_memories[:top_n]
    
    def describe_visual_content(self, image_path):
        """
        Create a descriptive summary of an image based on object detection.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Dictionary with description and detected objects
        """
        # Detect objects in the image
        _, _, detected_objects = self.detect_objects_in_image(image_path)
        
        if not detected_objects:
            return {
                "description": "No recognizable objects detected in this image.",
                "objects": []
            }
            
        # Generate a natural language description
        object_counts = {}
        for obj in detected_objects:
            label = obj["label"]
            count = obj["count"]
            object_counts[label] = count
            
        # Create a readable description
        description_parts = []
        
        # Include up to 5 top objects in the description
        top_objects = sorted(object_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        for label, count in top_objects:
            if count == 1:
                description_parts.append(f"a {label}")
            else:
                description_parts.append(f"{count} {label}s")
                
        if description_parts:
            if len(description_parts) == 1:
                description = f"This image contains {description_parts[0]}."
            elif len(description_parts) == 2:
                description = f"This image contains {description_parts[0]} and {description_parts[1]}."
            else:
                last_part = description_parts.pop()
                description = f"This image contains {', '.join(description_parts)}, and {last_part}."
        else:
            description = "This image contains various elements that could not be specifically identified."
            
        return {
            "description": description,
            "objects": detected_objects
        }
    
    def generate_simple_narrative(self, memories):
        """Generate a simple narrative from a set of memories."""
        if not memories:
            return "No memories found to generate a narrative."
        
        # Sort by date
        sorted_memories = sorted(memories, key=lambda m: m.get('date', ''))
        
        # Create simple narrative text
        narrative_parts = []
        for memory in sorted_memories:
            part = f"On {memory.get('date', 'an unknown date')}, "
            part += f"I was in {memory.get('location', 'an unknown place')}. "
            if memory.get('description'):
                part += memory.get('description')
                
            narrative_parts.append(part)
            
        # Join all parts
        return " ".join(narrative_parts)
    
    def sort_memories_by_relevance(self, memories, target_weight=5.0):
        """
        Sort memories by relevance and select ones until their total weight 
        reaches or exceeds the target weight.
        """
        if not memories:
            return []
            
        # Sort by weight in descending order
        sorted_memories = sorted(memories, key=lambda m: m.get('weight', 1.0), reverse=True)
        
        # Select memories until total weight reaches target
        selected_memories = []
        total_weight = 0.0
        
        for memory in sorted_memories:
            memory_weight = memory.get('weight', 1.0)
            selected_memories.append(memory)
            total_weight += memory_weight
            
            if total_weight >= target_weight:
                break
                
        return selected_memories