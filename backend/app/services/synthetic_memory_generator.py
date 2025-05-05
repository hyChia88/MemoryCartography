# backend/app/services/synthetic_memory_generator.py

import json
import os
import random
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image
import openai
from dotenv import load_dotenv
import torch
from torchvision.models import resnet50
import torchvision.transforms as T

load_dotenv()  # Load environment variables from .env file

class SyntheticMemoryGenerator:
    """Generate synthetic memory narratives based on real memories and spatial data."""
    
    def __init__(self):
        """Initialize the synthetic memory generator with visual analysis capabilities."""
        # Get OpenAI API key from environment variables
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            self.openai_client = openai.OpenAI(api_key=openai_api_key)
            self.openai_available = True
        else:
            self.openai_available = False
            print("WARNING: OpenAI API key not found. Synthetic memories will use fallback method.")
            
        # Initialize visual feature extraction capabilities
        self.image_model = self._initialize_image_model()
        if self.image_model:
            self.transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    
    def _initialize_image_model(self):
        """Initialize ResNet model for visual feature extraction."""
        try:
            model = resnet50(pretrained=True)
            model.eval()  # Set to evaluation mode
            return model
        except Exception as e:
            print(f"Error initializing image model: {e}")
            return None
            
    def extract_image_features(self, image_path):
        """Extract feature vector from an image for similarity comparison."""
        if not self.image_model:
            return None
            
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
    
    def analyze_floor_plan(self, floor_plan_path):
        """Extract architectural features from a floor plan image."""
        # This is a simplified placeholder - in a real system you would use
        # specialized architectural feature extraction models
        features = self.extract_image_features(floor_plan_path)
        
        if features is None:
            return {
                "success": False,
                "message": "Failed to extract features from floor plan"
            }
            
        # In a real system, this would detect rooms, doors, windows, etc.
        # For now, we'll return a simplified analysis
        return {
            "success": True,
            "feature_vector": features.tolist(),
            "estimated_rooms": 4,  # Placeholder values
            "estimated_area": 120,  # Square meters
            "detected_elements": [
                {"type": "room", "label": "living_room", "confidence": 0.92},
                {"type": "room", "label": "kitchen", "confidence": 0.89},
                {"type": "door", "count": 5, "confidence": 0.95},
                {"type": "window", "count": 6, "confidence": 0.91}
            ]
        }
        
    def find_similar_floor_plans(self, query_plan_path, dataset_paths, top_n=5):
        """Find similar floor plans based on visual features."""
        query_features = self.extract_image_features(query_plan_path)
        if query_features is None:
            return []
            
        results = []
        for path in dataset_paths:
            features = self.extract_image_features(path)
            if features is not None:
                # Calculate cosine similarity
                similarity = np.dot(query_features, features) / (
                    np.linalg.norm(query_features) * np.linalg.norm(features)
                )
                results.append((path, float(similarity)))
                
        # Sort by similarity (highest first)
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_n]
    
    def generate_memory_narrative(self, memories: List[Dict[str, Any]], search_term: str = None) -> Dict[str, Any]:
        """
        Generate a synthetic memory narrative based on a list of memories.
        
        Args:
            memories: List of memory objects with at least title, location, date, description, and keywords
            search_term: Optional search term that triggered the memory retrieval
            
        Returns:
            Dict containing synthetic text and other metadata
        """
        if not memories:
            return {
                "text": "No memories found to generate a narrative.",
                "keywords": [],
                "highlighted_terms": []
            }
        
        # Use OpenAI for generating the narrative if available
        if self.openai_available:
            return self._generate_with_openai(memories, search_term)
        else:
            return self._generate_fallback(memories, search_term)
    
    def generate_architectural_narrative(self, floor_plan_data, spatial_memories=None):
        """
        Generate a narrative description of architectural space.
        
        Args:
            floor_plan_data: Data extracted from floor plan analysis
            spatial_memories: Optional related spatial memories
            
        Returns:
            Dict containing narrative and metadata about the architectural space
        """
        if not floor_plan_data:
            return {
                "text": "No floor plan data provided for analysis.",
                "keywords": [],
                "spatial_elements": []
            }
            
        # Use OpenAI if available for rich architectural descriptions
        if self.openai_available:
            return self._generate_architectural_description_with_openai(floor_plan_data, spatial_memories)
        else:
            return self._generate_architectural_description_fallback(floor_plan_data, spatial_memories)
    
    def _generate_with_openai(self, memories: List[Dict[str, Any]], search_term: Optional[str]) -> Dict[str, Any]:
        """Generate synthetic memory using OpenAI."""
        try:
            # Sort memories by date
            memories_by_date = sorted(memories, key=lambda m: m.get('date', ''))
            
            # Create a list of memory descriptions for the prompt
            memory_descriptions = []
            all_keywords = []
            
            for i, memory in enumerate(memories_by_date):
                desc = f"Memory {i+1}: {memory.get('date', 'Unknown date')} - {memory.get('location', 'Unknown location')}\n"
                desc += f"Title: {memory.get('title', 'Untitled')}\n"
                desc += f"Description: {memory.get('description', '')}\n"
                
                keywords = memory.get('keywords', [])
                if keywords:
                    desc += f"Keywords: {', '.join(keywords)}\n"
                    all_keywords.extend(keywords)
                
                memory_descriptions.append(desc)
            
            # Create the prompt for OpenAI - Explicitly mention JSON to resolve the API error
            system_prompt = """You are a spatial memory synthesizer that creates natural, descriptive narratives from memory fragments. 
            Your task is to weave together multiple spatial memories into a coherent narrative that sounds like someone reminiscing about architectural spaces they've experienced.
            
            IMPORTANT: Respond ONLY in the following strict JSON format:
            {
              "narrative": "A diary-like narrative (3-5 sentences) about spatial experiences",
              "keywords": ["keyword1", "keyword2", ...],
              "highlighted_terms": ["term1", "term2", ...],
              "spatial_elements": ["element1", "element2", ...]
            }
            
            Guidelines for generating the narrative:
            1. Write in first person, as if these are your personal memories of spaces
            2. Focus on architectural and spatial qualities (light, proportion, materials, etc.)
            3. Mention how the spaces made you feel and how you interacted with them
            4. Make connections between different spatial experiences when possible
            5. Use a natural, conversational tone
            6. Highlight important architectural elements using *asterisks*
            7. Keep the narrative concise (3-5 sentences)
            8. Include a list of spatial elements mentioned (rooms, windows, materials, etc.)"""
            
            memories_text = "\n\n".join(memory_descriptions)
            
            if search_term:
                user_prompt = f"Generate a synthetic spatial memory narrative based on these related memories. The search term that triggered these memories was '{search_term}':\n\n{memories_text}"
            else:
                user_prompt = f"Generate a synthetic spatial memory narrative based on these related memories:\n\n{memories_text}"
            
            # Make the API call
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",  # Use an appropriate model
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            
            # Process the response
            if response.choices and response.choices[0].message.content:
                try:
                    result = json.loads(response.choices[0].message.content)
                    
                    # Extract the narrative and keywords
                    narrative = result.get("narrative", "")
                    keywords = result.get("keywords", [])
                    highlighted_terms = result.get("highlighted_terms", [])
                    spatial_elements = result.get("spatial_elements", [])
                    
                    return {
                        "text": narrative,
                        "keywords": keywords,
                        "highlighted_terms": highlighted_terms,
                        "spatial_elements": spatial_elements,
                        "source_memories": [m["id"] for m in memories]
                    }
                except json.JSONDecodeError:
                    print("Error parsing OpenAI response")
            
            # Fallback if OpenAI response processing fails
            return self._generate_fallback(memories, search_term)
            
        except Exception as e:
            print(f"Error generating memory with OpenAI: {e}")
            return self._generate_fallback(memories, search_term)
    
    def _generate_architectural_description_with_openai(self, floor_plan_data, spatial_memories=None):
        """Generate architectural description using OpenAI."""
        try:
            # Prepare floor plan data for the prompt
            floor_plan_info = json.dumps(floor_plan_data, indent=2)
            
            # Add spatial memories if available
            memories_text = ""
            if spatial_memories and len(spatial_memories) > 0:
                memory_descriptions = []
                for i, memory in enumerate(spatial_memories):
                    desc = f"Related Memory {i+1}: {memory.get('title', 'Untitled')}\n"
                    desc += f"Description: {memory.get('description', '')}\n"
                    memory_descriptions.append(desc)
                memories_text = "\n\n".join(memory_descriptions)
            
            # Create the prompt
            system_prompt = """You are an architectural analyst that creates detailed descriptions of spaces based on floor plan data.
            Your task is to generate an insightful architectural description highlighting key spatial features.
            
            IMPORTANT: Respond ONLY in the following strict JSON format:
            {
              "description": "An architectural description (3-5 sentences)",
              "spatial_qualities": ["quality1", "quality2", ...],
              "design_elements": ["element1", "element2", ...],
              "suggested_improvements": ["improvement1", "improvement2", ...]
            }
            
            Guidelines for generating the description:
            1. Focus on spatial organization, flow, and proportions
            2. Discuss natural light and views when information is available
            3. Comment on the relationship between spaces
            4. Highlight notable architectural features
            5. Include suggestions for potential improvements
            6. Keep the description objective and professional"""
            
            user_prompt = f"Generate an architectural description based on this floor plan data:\n\n{floor_plan_info}"
            
            if memories_text:
                user_prompt += f"\n\nConsider these related spatial memories as context:\n\n{memories_text}"
            
            # Make the API call
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            
            # Process the response
            if response.choices and response.choices[0].message.content:
                try:
                    result = json.loads(response.choices[0].message.content)
                    
                    return {
                        "text": result.get("description", ""),
                        "spatial_qualities": result.get("spatial_qualities", []),
                        "design_elements": result.get("design_elements", []),
                        "suggested_improvements": result.get("suggested_improvements", [])
                    }
                except json.JSONDecodeError:
                    print("Error parsing OpenAI architectural description response")
            
            # Fallback if OpenAI fails
            return self._generate_architectural_description_fallback(floor_plan_data, spatial_memories)
            
        except Exception as e:
            print(f"Error generating architectural description with OpenAI: {e}")
            return self._generate_architectural_description_fallback(floor_plan_data, spatial_memories)
    
    def _generate_architectural_description_fallback(self, floor_plan_data, spatial_memories=None):
        """Generate a simple architectural description without OpenAI."""
        # Extract data for description
        num_rooms = floor_plan_data.get("estimated_rooms", 0)
        area = floor_plan_data.get("estimated_area", 0)
        elements = floor_plan_data.get("detected_elements", [])
        
        # Create simple description
        text = f"This floor plan appears to be a {area} square meter space with approximately {num_rooms} rooms. "
        
        # Add details about detected elements
        room_types = [e["label"] for e in elements if e.get("type") == "room"]
        if room_types:
            text += f"The layout includes {', '.join(room_types)}. "
        
        # Count doors and windows
        doors = next((e for e in elements if e.get("type") == "door"), {})
        windows = next((e for e in elements if e.get("type") == "window"), {})
        
        if doors.get("count", 0) > 0:
            text += f"There are {doors.get('count')} doors. "
        
        if windows.get("count", 0) > 0:
            text += f"The space has {windows.get('count')} windows providing natural light. "
        
        # Add general design assessment
        design_qualities = ["functional", "compact", "spacious", "well-proportioned", "open"]
        selected_quality = random.choice(design_qualities)
        text += f"Overall, the design appears to be {selected_quality}."
        
        return {
            "text": text,
            "spatial_qualities": [selected_quality],
            "design_elements": [e["type"] for e in elements],
            "suggested_improvements": ["Consider improving natural light", "Optimize circulation paths"]
        }
    
    def _generate_fallback(self, memories: List[Dict[str, Any]], search_term: Optional[str]) -> Dict[str, Any]:
        """Generate synthetic memory using a simple fallback method."""
        # Sort memories by date
        sorted_memories = sorted(memories, key=lambda m: m.get('date', ''))
        
        # Collect metadata from memories
        dates = [m.get('date', '') for m in sorted_memories]
        titles = [m.get('title', '') for m in sorted_memories]
        locations = [m.get('location', '') for m in sorted_memories]
        descriptions = [m.get('description', '') for m in sorted_memories]
        
        # Collect all keywords
        all_keywords = []
        for memory in sorted_memories:
            all_keywords.extend(memory.get('keywords', []))
        
        # Get unique keywords
        unique_keywords = list(set(all_keywords))
        
        # Generate text based on the memories
        text = ""
        
        # Add introduction
        if len(sorted_memories) > 0:
            earliest_date = dates[0] if dates[0] else "an unknown date"
            latest_date = dates[-1] if dates[-1] else "recently"
            
            if earliest_date == latest_date:
                text += f"On {earliest_date}, "
            else:
                text += f"From {earliest_date} to {latest_date}, "
            
            # Add locations
            unique_locations = list(set(filter(None, locations)))
            if len(unique_locations) == 1:
                text += f"I experienced {titles[0]} in {unique_locations[0]}. "
            elif len(unique_locations) > 1:
                text += f"I explored spaces in {', '.join(unique_locations[:3])}. "
            else:
                text += f"I experienced {titles[0]}. "
        
        # Add selected details from descriptions
        if descriptions:
            if len(descriptions) == 1:
                # Use the single description
                text += descriptions[0] + " "
            else:
                # Combine elements from different descriptions
                elements = []
                for desc in descriptions[:3]:  # Use up to 3 descriptions
                    sentences = desc.split('. ')
                    if sentences:
                        elements.append(sentences[0])
                
                if elements:
                    text += " ".join(elements) + ". "
        
        # Add spatial reflection with keywords
        important_keywords = sorted(set(all_keywords), key=all_keywords.count, reverse=True)[:5]
        
        spatial_qualities = ["spacious", "bright", "cozy", "minimalist", "ornate", "balanced", "functional"]
        
        if important_keywords:
            text += f"The *{important_keywords[0]}* quality of these spaces created an atmosphere of *{random.choice(spatial_qualities)}*."
        else:
            text += f"These spaces had qualities of *{random.choice(spatial_qualities)}* and *{random.choice(spatial_qualities)}*."
        
        # Identify terms to highlight 
        highlighted_terms = []
        for keyword in important_keywords[:3]:  # Highlight up to 3 top keywords
            if keyword in text:
                highlighted_terms.append(keyword)
        
        # Extract spatial elements from memories
        spatial_elements = []
        architectural_keywords = ["room", "window", "door", "ceiling", "floor", "wall", "light", "space"]
        
        for keyword in all_keywords:
            if any(arch_term in keyword.lower() for arch_term in architectural_keywords):
                spatial_elements.append(keyword)
        
        # If search term is provided, add it to highlighted terms
        if search_term and search_term.lower() in text.lower():
            highlighted_terms.append(search_term)
        
        return {
            "text": text,
            "keywords": important_keywords,
            "highlighted_terms": highlighted_terms,
            "spatial_elements": spatial_elements[:5],
            "source_memories": [m["id"] for m in sorted_memories]
        }


# Create a singleton instance
_generator = None

def get_synthetic_memory_generator():
    global _generator
    if _generator is None:
        _generator = SyntheticMemoryGenerator()
    return _generator