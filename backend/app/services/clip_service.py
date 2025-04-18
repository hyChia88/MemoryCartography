import torch
import clip
from PIL import Image
import numpy as np
import os
import json
from pathlib import Path

class ClipService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading CLIP model on {self.device}...")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        
    def encode_image(self, image_path):
        """Encode an image to a feature vector using CLIP."""
        try:
            # Check if path is a string or a PIL Image
            if isinstance(image_path, str):
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image not found: {image_path}")
                image = Image.open(image_path).convert("RGB")
            else:
                # Assume it's a PIL image
                image = image_path.convert("RGB")
                
            # Preprocess and encode
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                
            # Return normalized features
            return image_features.cpu().numpy()
        except Exception as e:
            print(f"Error encoding image: {e}")
            return None
    
    def encode_text(self, text):
        """Encode text to a feature vector using CLIP."""
        try:
            text_input = clip.tokenize([text]).to(self.device)
            with torch.no_grad():
                text_features = self.model.encode_text(text_input)
            return text_features.cpu().numpy()
        except Exception as e:
            print(f"Error encoding text: {e}")
            return None
    
    def extract_keywords(self, image_path, candidate_keywords=None):
        """Extract relevant keywords from an image."""
        if candidate_keywords is None:
            # Default candidate keywords for common scene elements
            candidate_keywords = [
                "beach", "mountain", "city", "lake", "forest", 
                "sunset", "building", "people", "food", "art",
                "history", "nature", "architecture", "water", "sky",
                "street", "park", "market", "cafe", "restaurant",
                "museum", "monument", "bridge", "river", "garden",
                "tower", "castle", "temple", "church", "house",
                "road", "train", "car", "bicycle", "boat",
                "airport", "station", "hotel", "night", "day",
                "spring", "summer", "fall", "winter", "snow",
                "rain", "fog", "clouds", "clear sky", "storm"
            ]
        
        # Encode the image
        image_features = self.encode_image(image_path)
        if image_features is None:
            return []
        
        # Encode all candidate keywords
        text_tokens = clip.tokenize(candidate_keywords).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
        
        # Calculate similarities
        image_features = torch.from_numpy(image_features).to(self.device)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        similarity = similarity.cpu().numpy()[0]
        
        # Get top keywords based on similarity
        top_indices = similarity.argsort()[-10:][::-1]  # Top 10 keywords
        top_keywords = [(candidate_keywords[idx], float(similarity[idx])) for idx in top_indices]
        
        return top_keywords
    
    def batch_process_images(self, image_dir, output_file=None, metadata_file=None):
        """Process all images in a directory and save their features."""
        image_dir = Path(image_dir)
        results = []
        
        # Load metadata if provided
        metadata = {}
        if metadata_file and os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        
        # Process all images
        for img_path in image_dir.glob("**/*.jpg") + image_dir.glob("**/*.jpeg") + image_dir.glob("**/*.png"):
            rel_path = str(img_path.relative_to(image_dir))
            
            # Extract features
            features = self.encode_image(str(img_path))
            if features is None:
                continue
                
            # Extract keywords
            keywords = self.extract_keywords(str(img_path))
            keywords = [k[0] for k in keywords]  # Just keep the keyword text
            
            # Get metadata if available
            image_metadata = metadata.get(rel_path, {})
            location = image_metadata.get('location', '')
            date = image_metadata.get('date', '')
            
            results.append({
                'path': rel_path,
                'features': features.tolist()[0],
                'keywords': keywords,
                'location': location,
                'date': date
            })
            
            print(f"Processed {rel_path} - Keywords: {keywords[:5]}")
        
        # Save results
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f)
        
        return results

# Singleton instance
_instance = None

def get_clip_service():
    global _instance
    if _instance is None:
        _instance = ClipService()
    return _instance