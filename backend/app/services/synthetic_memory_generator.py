# backend/app/services/synthetic_memory_generator.py

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
                "highlighted_terms": []
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
            
            for i, memory in enumerate(memories_by_date):
                desc = f"Memory {i+1}: {memory.get('date', 'Unknown date')} - {memory.get('location', 'Unknown location')}\n"
                desc += f"Title: {memory.get('title', 'Untitled')}\n"
                desc += f"Description: {memory.get('description', '')}\n"
                
                keywords = memory.get('keywords', [])
                if keywords:
                    desc += f"Keywords: {', '.join(keywords)}\n"
                    all_keywords.extend(keywords)
                
                memory_descriptions.append(desc)
            
            # Create the prompt for OpenAI
            system_prompt = """You are a personal memory synthesizer that creates natural, diary-like narratives from memory fragments. 
            Your task is to weave together multiple memories into a coherent narrative that sounds like someone reminiscing.
            
            Follow these guidelines:
            1. Write in first person, as if these are your personal memories
            2. Maintain chronological order when applicable
            3. Focus on sensory details and emotions
            4. Make connections between different memories when possible
            5. Use a natural, conversational tone
            6. Keep the narrative focused on the memories provided
            7. Highlight important keywords in the narrative with *asterisks*
            8. The text should be 3-5 sentences - concise but evocative
            
            The output should include:
            1. A diary-like narrative (3-5 sentences)
            2. A list of the 5-10 most important keywords that were used in the narrative
            3. A list of terms that were highlighted with asterisks"""
            
            memories_text = "\n\n".join(memory_descriptions)
            
            if search_term:
                user_prompt = f"Generate a synthetic memory narrative based on these related memories. The search term that triggered these memories was '{search_term}':\n\n{memories_text}"
            else:
                user_prompt = f"Generate a synthetic memory narrative based on these related memories:\n\n{memories_text}"
            
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
                    
                    return {
                        "text": narrative,
                        "keywords": keywords,
                        "highlighted_terms": highlighted_terms,
                        "source_memories": [m["id"] for m in memories]
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
                text += f"I traveled between {', '.join(unique_locations[:3])}. "
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
        
        # Add emotional reflection with keywords
        important_keywords = sorted(set(all_keywords), key=all_keywords.count, reverse=True)[:5]
        if important_keywords:
            text += f"These memories evoke feelings of *{important_keywords[0]}* and *{important_keywords[1] if len(important_keywords) > 1 else 'nostalgia'}*."
        else:
            emotions = ["joy", "nostalgia", "awe", "excitement", "peace", "wonder"]
            text += f"These memories evoke feelings of *{random.choice(emotions)}* and *{random.choice(emotions)}*."
        
        # Identify terms to highlight (we'll use asterisks in fallback method)
        highlighted_terms = []
        for keyword in important_keywords[:3]:  # Highlight up to 3 top keywords
            if keyword in text:
                highlighted_terms.append(keyword)
        
        # If search term is provided, add it to highlighted terms
        if search_term and search_term.lower() in text.lower():
            highlighted_terms.append(search_term)
        
        return {
            "text": text,
            "keywords": important_keywords,
            "highlighted_terms": highlighted_terms,
            "source_memories": [m["id"] for m in sorted_memories]
        }


# Create a singleton instance
_generator = None

def get_synthetic_memory_generator():
    global _generator
    if _generator is None:
        _generator = SyntheticMemoryGenerator()
    return _generator