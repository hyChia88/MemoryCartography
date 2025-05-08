# synthetic_memory_generator.py
import json
import os
import random
from datetime import datetime
from typing import List, Dict, Any, Optional
import openai
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

class SyntheticMemoryGenerator:
    """Generate synthetic memory narratives based on real memories."""
    
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
                "highlighted_terms": [],
                "source_memories": []
            }
        
        # Use OpenAI for generating the narrative if available
        if self.openai_available:
            return self._generate_with_openai(memories, search_term)
        else:
            return self._generate_fallback(memories, search_term)
    
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
                    
                    try:
                        detected_objects = json.loads(memory.get('detected_objects'))
                        for obj in detected_objects:
                            if isinstance(obj, str):
                                obj_list.append(obj)
                    except (json.JSONDecodeError, TypeError):
                        pass
                    
                    if obj_list:
                        obj_desc += ", ".join(obj_list)
                        desc += obj_desc + "\n"
                        detected_objects_info.extend(obj_list)
                
                keywords = memory.get('keywords', [])
                if keywords:
                    desc += f"Keywords: {', '.join(keywords)}\n"
                    all_keywords.extend(keywords)
                
                memory_descriptions.append(desc)
            
            # Create system prompt for OpenAI
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
            1. Write in first person, as if these are personal memories of spaces
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
                try:
                    detected_objects = json.loads(memory.get('detected_objects'))
                    if isinstance(detected_objects, list):
                        for obj in detected_objects:
                            if isinstance(obj, str) and obj not in all_objects:
                                all_objects.append(obj)
                except (json.JSONDecodeError, TypeError):
                    pass
        
        # Generate narrative text
        text = ""
        
        # Add introduction with date and location
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
        
        # Add reflection with keywords
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
        
        # Extract spatial elements
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

# Singleton instance
_generator = None

def get_synthetic_memory_generator():
    """Get or create the singleton instance of SyntheticMemoryGenerator."""
    global _generator
    if _generator is None:
        _generator = SyntheticMemoryGenerator()
    return _generator