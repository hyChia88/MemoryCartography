# app/services/synthetic_memory_generator.py
import json
import os
import random
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('logs/synthetic_memory.log', mode='a'),
        logging.StreamHandler()
    ]
)

class SyntheticMemoryGenerator:
    """Generate synthetic memory narratives based on real memories without using external APIs."""
    
    def __init__(self):
        """Initialize the synthetic memory generator."""
        logging.info("Initializing synthetic memory generator")
        
        # Load vocabulary for more varied descriptions
        self.spatial_terms = [
            "spacious", "intimate", "cozy", "vast", "cramped", "airy", 
            "bright", "dim", "illuminated", "shadowy", "warm", "cool", 
            "expansive", "confined", "open", "enclosed", "sunlit", "shaded"
        ]
        
        self.emotion_terms = [
            "peaceful", "calming", "energizing", "inspiring", "nostalgic", 
            "tranquil", "vibrant", "serene", "melancholic", "joyful",
            "overwhelming", "inviting", "mysterious", "comfortable", "soothing"
        ]
        
        self.architectural_elements = [
            "walls", "ceiling", "floor", "windows", "doors", "beams", 
            "columns", "arches", "stairs", "corners", "balcony", "terrace",
            "roof", "facade", "entrance", "exit", "hallway", "room"
        ]
        
        self.spatial_qualities = [
            "proportion", "balance", "symmetry", "asymmetry", "rhythm", 
            "texture", "color", "light", "shadow", "scale", "perspective",
            "depth", "height", "width", "contrast", "harmony", "pattern"
        ]
        
        self.sensory_words = [
            "echoing", "silent", "bustling", "quiet", "fragrant", "musty",
            "fresh", "stale", "bright", "dim", "warm", "cool", "rough", 
            "smooth", "hard", "soft", "spacious", "cramped", "open", "closed"
        ]
        
    def _extract_keywords_and_objects(self, memories: List[Dict[str, Any]]) -> tuple:
        """
        Extract keywords and detected objects from memories.
        
        Args:
            memories: List of memory dictionaries
            
        Returns:
            Tuple of (all keywords, all objects)
        """
        all_keywords = []
        all_objects = []
        
        for memory in memories:
            # Extract keywords
            keywords = []
            if 'keywords' in memory:
                if isinstance(memory['keywords'], str):
                    try:
                        keywords = json.loads(memory['keywords'])
                    except json.JSONDecodeError:
                        keywords = []
                elif isinstance(memory['keywords'], list):
                    keywords = memory['keywords']
            
            all_keywords.extend(keywords)
            
            # Extract detected objects
            objects = []
            if 'detected_objects' in memory:
                if isinstance(memory['detected_objects'], str):
                    try:
                        objects = json.loads(memory['detected_objects'])
                    except json.JSONDecodeError:
                        objects = []
                elif isinstance(memory['detected_objects'], list):
                    objects = memory['detected_objects']
            
            all_objects.extend(objects)
        
        # Remove duplicates
        all_keywords = list(set(all_keywords))
        all_objects = list(set(all_objects))
        
        return all_keywords, all_objects
    
    def _extract_locations_and_dates(self, memories: List[Dict[str, Any]]) -> tuple:
        """
        Extract locations and dates from memories.
        
        Args:
            memories: List of memory dictionaries
            
        Returns:
            Tuple of (unique locations, sorted dates)
        """
        locations = []
        dates = []
        
        for memory in memories:
            # Extract location
            location = memory.get('location', 'Unknown Location')
            if location and location != 'Unknown Location':
                # Just use the first part of the location (before any commas)
                location = location.split(',')[0].strip()
                locations.append(location)
            
            # Extract date
            date = memory.get('date', '')
            if date:
                dates.append(date)
        
        # Remove duplicates from locations
        unique_locations = list(set(locations))
        
        # Sort dates
        sorted_dates = sorted(dates) if dates else []
        
        return unique_locations, sorted_dates
    
    def _find_architectural_objects(self, all_objects: List[str]) -> List[str]:
        """
        Find architectural objects from a list of detected objects.
        
        Args:
            all_objects: List of detected object names
            
        Returns:
            List of architectural objects
        """
        architectural_terms = [
            "room", "building", "house", "wall", "ceiling", "floor", "window", 
            "door", "roof", "column", "beam", "arch", "stair", "balcony", 
            "terrace", "facade", "entrance", "exit", "hallway", "lobby", 
            "corridor", "structure", "interior", "exterior", "space"
        ]
        
        architectural_objects = []
        
        for obj in all_objects:
            obj_lower = obj.lower()
            if any(term in obj_lower for term in architectural_terms):
                architectural_objects.append(obj)
        
        return architectural_objects
    
    def generate_memory_narrative(self, memories: List[Dict[str, Any]], search_term: Optional[str] = None) -> Dict[str, Any]:
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
        
        # Extract metadata from memories
        all_keywords, all_objects = self._extract_keywords_and_objects(memories)
        locations, dates = self._extract_locations_and_dates(memories)
        
        # Sort memories by date
        sorted_memories = sorted(memories, key=lambda m: m.get('date', ''))
        
        # Find architectural objects
        architectural_objects = self._find_architectural_objects(all_objects)
        
        # Generate narrative text
        text = self._generate_narrative_text(
            sorted_memories,
            locations,
            dates,
            all_keywords,
            all_objects,
            architectural_objects,
            search_term
        )
        
        # Select important keywords from the narrative
        narrative_keywords = self._select_important_keywords(
            all_keywords, 
            all_objects,
            locations,
            search_term
        )
        
        # Identify terms to highlight in the narrative
        highlighted_terms = self._find_highlight_terms(
            text,
            narrative_keywords,
            all_objects,
            search_term
        )
        
        # Return the complete result
        return {
            "text": text,
            "keywords": narrative_keywords,
            "highlighted_terms": highlighted_terms,
            "source_memories": [m["id"] for m in sorted_memories if "id" in m]
        }
    
    def _generate_narrative_text(
        self,
        sorted_memories: List[Dict[str, Any]],
        locations: List[str],
        dates: List[str],
        all_keywords: List[str],
        all_objects: List[str],
        architectural_objects: List[str],
        search_term: Optional[str]
    ) -> str:
        """
        Generate narrative text based on the memory data.
        
        Args:
            Various extracted memory data
            
        Returns:
            Narrative text string
        """
        # Start with time and location info
        text = ""
        
        # Add introduction with date and location
        if dates:
            earliest_date = dates[0] if dates else "an unknown date"
            latest_date = dates[-1] if dates and len(dates) > 1 else earliest_date
            
            if earliest_date == latest_date:
                text += f"On {earliest_date}, "
            else:
                text += f"From {earliest_date} to {latest_date}, "
        else:
            text += "During my exploration, "
        
        # Add locations
        if locations:
            if len(locations) == 1:
                text += f"I experienced *{sorted_memories[0].get('title', 'a space')}* in *{locations[0]}*. "
            elif len(locations) > 1:
                text += f"I explored spaces in {', '.join(['*' + loc + '*' for loc in locations[:3]])}. "
        else:
            text += f"I experienced *{sorted_memories[0].get('title', 'a space')}*. "
        
        # Add object detection information
        if all_objects:
            # Select some interesting objects (prioritize architectural elements)
            selected_objects = architectural_objects[:3] if architectural_objects else []
            
            # If we need more objects, add some from the general list
            if len(selected_objects) < 3:
                for obj in all_objects:
                    if obj not in selected_objects and len(selected_objects) < 3:
                        selected_objects.append(obj)
            
            if len(selected_objects) == 1:
                text += f"The space contained a *{selected_objects[0]}* that caught my attention. "
            elif len(selected_objects) == 2:
                text += f"I noticed both a *{selected_objects[0]}* and a *{selected_objects[1]}* within the environment. "
            elif len(selected_objects) > 2:
                text += f"The area featured *{selected_objects[0]}*, *{selected_objects[1]}*, and *{selected_objects[2]}* that defined the space. "
        
        # Add spatial qualities
        spatial_quality = random.choice(self.spatial_terms)
        emotional_quality = random.choice(self.emotion_terms)
        
        text += f"The space felt *{spatial_quality}* and *{emotional_quality}*. "
        
        # Add sensory experience
        sensory_word = random.choice(self.sensory_words)
        text += f"I remember how *{sensory_word}* the environment was as I moved through it. "
        
        # If search term was provided, try to incorporate it
        if search_term:
            # Only add this if the search term isn't already in the text
            if search_term.lower() not in text.lower():
                text += f"This memory especially connects to *{search_term}* through its spatial qualities. "
        
        return text
    
    def _select_important_keywords(
        self, 
        all_keywords: List[str], 
        all_objects: List[str],
        locations: List[str],
        search_term: Optional[str]
    ) -> List[str]:
        """
        Select the most important keywords for the narrative.
        
        Args:
            all_keywords: All keywords from memories
            all_objects: All detected objects
            locations: All locations
            search_term: Optional search term
            
        Returns:
            List of important keywords
        """
        important_keywords = []
        
        # Add search term if provided
        if search_term:
            important_keywords.append(search_term)
        
        # Add location keywords
        important_keywords.extend(locations[:2])
        
        # Add some regular keywords
        for keyword in all_keywords:
            if len(important_keywords) < 5 and keyword not in important_keywords:
                important_keywords.append(keyword)
        
        # Add some objects
        for obj in all_objects:
            if len(important_keywords) < 7 and obj not in important_keywords:
                important_keywords.append(obj)
        
        # Return unique keywords
        return list(set(important_keywords))
    
    def _find_highlight_terms(
        self,
        text: str,
        narrative_keywords: List[str],
        all_objects: List[str],
        search_term: Optional[str]
    ) -> List[str]:
        """
        Find terms that should be highlighted in the narrative.
        
        Args:
            text: Narrative text
            narrative_keywords: Selected keywords
            all_objects: All detected objects
            search_term: Optional search term
            
        Returns:
            List of terms to highlight
        """
        # Extract terms surrounded by asterisks
        highlighted_terms = re.findall(r'\*(.*?)\*', text)
        
        # Add search term if it's in the text but not highlighted
        if search_term and search_term.lower() in text.lower() and search_term not in highlighted_terms:
            highlighted_terms.append(search_term)
        
        # Return unique terms
        return list(set(highlighted_terms))


# Singleton instance
_generator = None

def get_synthetic_memory_generator() -> SyntheticMemoryGenerator:
    """Get or create the singleton instance of SyntheticMemoryGenerator."""
    global _generator
    if _generator is None:
        _generator = SyntheticMemoryGenerator()
    return _generator