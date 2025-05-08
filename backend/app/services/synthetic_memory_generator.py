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

load_dotenv()  # Load environment variables from .env file

class SyntheticMemoryGenerator:
    """Generate synthetic memory narratives based on real memories and detected objects."""
    
    def __init__(self):
        """Initialize the synthetic memory generator."""
        # Get OpenAI API key from environment variables
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            self.openai_client = openai.OpenAI(api_key=openai_api_key)
            self.openai_available = True
        else:
            self.openai_available = False
            print("WARNING: OpenAI API key not found. Synthetic memories will use fallback method.")
    
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
    
    def generate_spatial_narrative_from_objects(self, detected_objects: List[Dict], location: str = "an unknown place", date: str = None) -> Dict[str, Any]:
        """
        Generate a narrative description based on detected objects in an image.
        
        Args:
            detected_objects: List of objects detected in the image
            location: Optional location information
            date: Optional date information
            
        Returns:
            Dict containing narrative text and metadata
        """
        if not detected_objects:
            return {
                "text": "No objects detected to generate a narrative.",
                "keywords": [],
                "spatial_elements": []
            }
            
        # Use OpenAI if available
        if self.openai_available:
            return self._generate_spatial_narrative_with_openai(detected_objects, location, date)
        else:
            return self._generate_spatial_narrative_fallback(detected_objects, location, date)
    
    def _generate_with_openai(self, memories: List[Dict[str, Any]], search_term: Optional[str]) -> Dict[str, Any]:
        """Generate synthetic memory using OpenAI."""
        try:
            # Sort memories by date
            memories_by_date = sorted(memories, key=lambda m: m.get('date', ''))
            
            # Create a list of memory descriptions for the prompt
            memory_descriptions = []
            all_keywords = []
            detected_objects_info = []
            
            for i, memory in enumerate(memories_by_date):
                desc = f"Memory {i+1}: {memory.get('date', 'Unknown date')} - {memory.get('location', 'Unknown location')}\n"
                desc += f"Title: {memory.get('title', 'Untitled')}\n"
                desc += f"Description: {memory.get('description', '')}\n"
                
                # Add object detection results if available
                if memory.get('detected_objects'):
                    obj_desc = "Detected objects: "
                    obj_list = []
                    
                    for obj in memory.get('detected_objects'):
                        obj_list.append(f"{obj.get('label')} (count: {obj.get('count')})")
                        
                    if obj_list:
                        obj_desc += ", ".join(obj_list)
                        desc += obj_desc + "\n"
                        detected_objects_info.extend(obj_list)
                
                keywords = memory.get('keywords', [])
                if keywords:
                    desc += f"Keywords: {', '.join(keywords)}\n"
                    all_keywords.extend(keywords)
                
                memory_descriptions.append(desc)
            
            # Create the prompt for OpenAI with explicit JSON format
            system_prompt = """You are a spatial memory synthesizer that creates natural, descriptive narratives from memory fragments. 
            Your task is to weave together multiple spatial memories into a coherent narrative that sounds like someone reminiscing about spaces they've experienced.
            
            IMPORTANT: Respond ONLY in the following strict JSON format:
            {
              "narrative": "A diary-like narrative (3-5 sentences) about spatial experiences",
              "keywords": ["keyword1", "keyword2", ...],
              "highlighted_terms": ["term1", "term2", ...],
              "spatial_elements": ["element1", "element2", ...]
            }
            
            Guidelines for generating the narrative:
            1. Write in first person, as if these are your personal memories of spaces
            2. Focus on the objects detected in the memories and their spatial relationships
            3. Mention how the spaces made you feel and how you interacted with them
            4. Make connections between different spatial experiences when possible
            5. Use a natural, conversational tone
            6. Highlight important objects or elements using *asterisks*
            7. Keep the narrative concise (3-5 sentences)
            8. Include a list of the spatial elements mentioned"""
            
            memories_text = "\n\n".join(memory_descriptions)
            
            if search_term:
                user_prompt = f"Generate a synthetic memory narrative based on these memories with detected objects. The search term that triggered these memories was '{search_term}':\n\n{memories_text}"
            else:
                user_prompt = f"Generate a synthetic memory narrative based on these memories with detected objects:\n\n{memories_text}"
            
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
                        "source_memories": [m["id"] for m in memories if "id" in m]
                    }
                except json.JSONDecodeError:
                    print("Error parsing OpenAI response")
            
            # Fallback if OpenAI response processing fails
            return self._generate_fallback(memories, search_term)
            
        except Exception as e:
            print(f"Error generating memory with OpenAI: {e}")
            return self._generate_fallback(memories, search_term)
    
    def _generate_spatial_narrative_with_openai(self, detected_objects: List[Dict], location: str, date: str) -> Dict[str, Any]:
        """Generate a spatial narrative from detected objects using OpenAI."""
        try:
            # Prepare object detection data
            object_descriptions = []
            
            for obj in detected_objects:
                label = obj.get('label', 'unknown')
                count = obj.get('count', 1)
                confidence = obj.get('confidence', 0.0)
                
                if count == 1:
                    object_descriptions.append(f"A {label} (confidence: {confidence:.2f})")
                else:
                    object_descriptions.append(f"{count} {label}s (confidence: {confidence:.2f})")
            
            # Create the prompt
            system_prompt = """You are a spatial analyst that creates descriptive narratives from detected objects in an image. 
            Your task is to create a coherent narrative that describes the spatial arrangement and relationships between objects.
            
            IMPORTANT: Respond ONLY in the following strict JSON format:
            {
              "narrative": "A descriptive narrative (3-5 sentences) about the spatial scene",
              "keywords": ["keyword1", "keyword2", ...],
              "spatial_elements": ["element1", "element2", ...],
              "spatial_relationships": ["relationship1", "relationship2", ...]
            }
            
            Guidelines for generating the narrative:
            1. Focus on the spatial relationships between detected objects
            2. Imagine how these objects might be arranged in a realistic scene
            3. Use specific spatial terms (above, below, beside, between, etc.)
            4. Describe potential architectural elements that might contain these objects
            5. Keep the narrative concise and focused on spatial qualities"""
            
            objects_text = "\n".join(object_descriptions)
            
            date_info = f" on {date}" if date else ""
            user_prompt = f"Generate a spatial narrative based on these objects detected in an image from {location}{date_info}:\n\n{objects_text}"
            
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
                        "text": result.get("narrative", ""),
                        "keywords": result.get("keywords", []),
                        "spatial_elements": result.get("spatial_elements", []),
                        "spatial_relationships": result.get("spatial_relationships", [])
                    }
                except json.JSONDecodeError:
                    print("Error parsing OpenAI spatial narrative response")
            
            # Fallback
            return self._generate_spatial_narrative_fallback(detected_objects, location, date)
            
        except Exception as e:
            print(f"Error generating spatial narrative with OpenAI: {e}")
            return self._generate_spatial_narrative_fallback(detected_objects, location, date)
    
    def _generate_spatial_narrative_fallback(self, detected_objects: List[Dict], location: str, date: str) -> Dict[str, Any]:
        """Generate a fallback spatial narrative from detected objects."""
        if not detected_objects:
            return {
                "text": "No objects were detected in this scene.",
                "keywords": [],
                "spatial_elements": []
            }
            
        # Count object types
        object_counts = {}
        for obj in detected_objects:
            label = obj.get('label', 'object')
            count = obj.get('count', 1)
            object_counts[label] = count
            
        # Create description parts
        description_parts = []
        
        # Add date and location
        date_str = f"On {date}, " if date else ""
        location_str = f"in {location}" if location != "an unknown place" else "in a space"
        
        intro = f"{date_str}I observed a scene {location_str} containing "
        
        # Add object descriptions
        object_phrases = []
        for label, count in object_counts.items():
            if count == 1:
                object_phrases.append(f"a *{label}*")
            else:
                object_phrases.append(f"{count} *{label}s*")
                
        if len(object_phrases) == 1:
            intro += f"{object_phrases[0]}. "
        elif len(object_phrases) == 2:
            intro += f"{object_phrases[0]} and {object_phrases[1]}. "
        else:
            last_phrase = object_phrases.pop()
            intro += f"{', '.join(object_phrases)}, and {last_phrase}. "
            
        description_parts.append(intro)
        
        # Add spatial relationships if we have multiple objects
        if len(object_counts) > 1:
            # Get object keys
            object_keys = list(object_counts.keys())
            
            # Create some imagined spatial relationships
            spatial_relationships = [
                "beside",
                "near",
                "surrounding",
                "in front of",
                "behind",
                "above",
                "below"
            ]
            
            # Generate a random relationship
            if len(object_keys) >= 2:
                obj1 = object_keys[0]
                obj2 = object_keys[1]
                relation = random.choice(spatial_relationships)
                
                description_parts.append(f"The {obj1} was {relation} the {obj2}, creating an interesting spatial dynamic. ")
                
        # Add a reflective statement about the space
        spatial_qualities = [
            "inviting",
            "structured",
            "balanced",
            "dynamic",
            "harmonious",
            "functional"
        ]
        
        quality = random.choice(spatial_qualities)
        description_parts.append(f"The arrangement felt *{quality}* and made an impression on my spatial memory.")
        
        # Combine parts into full narrative
        narrative = "".join(description_parts)
        
        # Extract keywords
        keywords = list(object_counts.keys())
        if quality:
            keywords.append(quality)
            
        # Extract spatial elements (all objects are spatial elements)
        spatial_elements = list(object_counts.keys())
        
        # Highlighted terms are the object labels
        highlighted_terms = [term for term in keywords if term in narrative]
        
        return {
            "text": narrative,
            "keywords": keywords,
            "highlighted_terms": highlighted_terms,
            "spatial_elements": spatial_elements
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
        
        # Collect all keywords and detected objects
        all_keywords = []
        all_objects = []
        
        for memory in sorted_memories:
            all_keywords.extend(memory.get('keywords', []))
            
            # Extract object information if available
            if memory.get('detected_objects'):
                for obj in memory.get('detected_objects'):
                    obj_label = obj.get('label', '')
                    if obj_label and obj_label not in all_objects:
                        all_objects.append(obj_label)
        
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
        
        # Add detected objects if available
        if all_objects:
            # Take up to 5 most interesting objects
            selected_objects = all_objects[:5]
            
            if len(selected_objects) == 1:
                text += f"I noticed a *{selected_objects[0]}* in the space. "
            elif len(selected_objects) == 2:
                text += f"The space contained a *{selected_objects[0]}* and a *{selected_objects[1]}*. "
            else:
                last_obj = selected_objects.pop()
                text += f"I observed various objects including {', '.join(['*' + obj + '*' for obj in selected_objects])}, and a *{last_obj}*. "
        
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
        
        # Add detected objects to highlighted terms
        for obj in all_objects[:3]:  # Add up to 3 top objects
            if obj in text and obj not in highlighted_terms:
                highlighted_terms.append(obj)
        
        # Extract spatial elements (combine keywords and objects)
        spatial_elements = []
        architectural_keywords = ["room", "window", "door", "ceiling", "floor", "wall", "light", "space"]
        
        # From keywords
        for keyword in all_keywords:
            if any(arch_term in keyword.lower() for arch_term in architectural_keywords):
                spatial_elements.append(keyword)
                
        # From detected objects
        for obj in all_objects:
            if any(arch_term in obj.lower() for arch_term in architectural_keywords):
                if obj not in spatial_elements:
                    spatial_elements.append(obj)
        
        # If search term is provided, add it to highlighted terms
        if search_term and search_term.lower() in text.lower():
            highlighted_terms.append(search_term)
        
        return {
            "text": text,
            "keywords": important_keywords + all_objects[:3],  # Combine keywords and top objects
            "highlighted_terms": highlighted_terms,
            "spatial_elements": spatial_elements[:5],
            "source_memories": [m["id"] for m in sorted_memories if "id" in m]
        }
        
# Create a singleton instance
_generator = None

def get_synthetic_memory_generator():
    global _generator
    if _generator is None:
        _generator = SyntheticMemoryGenerator()
    return _generator