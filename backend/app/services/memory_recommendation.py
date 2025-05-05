# memory_recommendation.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
from datetime import datetime
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.models import resnet50
import os

class MemoryRecommendationEngine:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        # Initialize ResNet model for visual feature extraction
        self.image_model = self._initialize_image_model()
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
            print(f"Error initializing image model: {e}")
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
    
    def find_similar_spatial_objects(self, query_object, dataset_objects, top_n=5):
        """Find similar architectural/spatial objects based on visual features."""
        # This method can be used to find similar furniture, architectural elements, etc.
        if not self.image_model:
            return []
            
        query_features = query_object.get('features')
        if not query_features:
            return []
            
        # Get features from all objects
        object_features = []
        valid_objects = []
        
        for obj in dataset_objects:
            features = obj.get('features')
            if features is not None:
                object_features.append(features)
                valid_objects.append(obj)
        
        if not object_features:
            return []
            
        # Convert to numpy arrays for similarity calculation
        query_features_array = np.array(query_features)
        object_features_array = np.vstack(object_features)
        
        # Calculate similarities
        similarities = cosine_similarity(query_features_array.reshape(1, -1), object_features_array)[0]
        
        # Sort by similarity and get top matches
        top_indices = similarities.argsort()[::-1][:top_n]
        
        # Return results with similarity scores
        results = [(valid_objects[i], similarities[i]) for i in top_indices]
        return results
    
    def generate_synthetic_narrative(self, memories):
        """Generate a simple synthetic narrative from memories."""
        if not memories:
            return "No memories found to generate a narrative."
        
        # Select top memories
        tol_weight = 0
        i = 0
        while i < len(memories) and tol_weight < 5.0:
            tol_weight += memories[i].get('weight', 1.0)
            i += 1

        selected_memories = memories[:i]
        
        # Create narrative text
        narrative_parts = []
        for memory in selected_memories:
            narrative_parts.append(
                f"On {memory.get('date', 'an unknown date')}, "
                f"I was in {memory.get('location', 'an unknown place')}. "
                f"{memory.get('description', '')}"
            )
        
        return " ".join(narrative_parts)
    
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
    
    def find_memories_by_spatial_features(self, query_spatial_features, all_memories, top_n=5):
        """Find memories with similar spatial features using both visual and textual information."""
        # If we have image features for the query
        if query_spatial_features.get('image_features') is not None:
            # First try to find visually similar spaces
            memories_with_images = [m for m in all_memories if m.get('image_features') is not None]
            if memories_with_images:
                query_features = np.array(query_spatial_features['image_features'])
                memory_features = np.vstack([np.array(m['image_features']) for m in memories_with_images])
                
                # Calculate similarities
                similarities = cosine_similarity(query_features.reshape(1, -1), memory_features)[0]
                
                # Sort by similarity
                top_indices = similarities.argsort()[::-1][:top_n]
                visual_matches = [memories_with_images[i] for i in top_indices]
                
                return visual_matches
        
        # Fallback to text-based similarity
        return self.find_similar_memories(query_spatial_features, all_memories, top_n)
    
    def process_architectural_elements(self, floor_plan_image, element_type=None):
        """Process architectural elements from a floor plan image."""
        
        return {
            "elements": [
                {"type": "room", "label": "living_room", "area": 24.5, "bounding_box": [100, 100, 300, 300]},
                {"type": "door", "width": 0.9, "bounding_box": [150, 300, 180, 310]},
                {"type": "window", "width": 1.2, "bounding_box": [250, 100, 290, 110]}
            ],
            "floor_plan_features": self.extract_image_features(floor_plan_image) if os.path.exists(floor_plan_image) else None
        }
    
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