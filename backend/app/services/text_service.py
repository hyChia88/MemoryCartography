import random
from typing import List, Dict, Any

def generate_narrative(memories: List[Dict[str, Any]], location: str, database_type: str = "user"):
    """Generate a narrative about the location based on available memories."""
    
    if not memories:
        return {
            "text": f"No memories found for {location}. Try searching for another location or add some memories first.",
            "keywords": [{"text": location, "type": "primary"}]
        }
    
    # Collect all keywords from memories
    all_keywords = []
    for memory in memories:
        all_keywords.extend(memory.get("keywords", []))
    
    # Remove duplicates while preserving order
    unique_keywords = []
    for kw in all_keywords:
        if kw not in unique_keywords:
            unique_keywords.append(kw)
    
    # Generate primary keywords (including location)
    primary_keywords = [location]
    
    # Add up to 4 more keywords from memories
    primary_keywords.extend(unique_keywords[:4])
    
    # Generate connected keywords (concepts associated with memories)
    connected_concepts = [
        "nostalgia", "joy", "reflection", "journey", "discovery", 
        "experience", "memory", "moment", "feeling", "impression"
    ]
    connected_keywords = random.sample(connected_concepts, min(3, len(connected_concepts)))
    
    # Generate narrative
    if database_type == "user":
        narrative = f"When I think of {location}, I recall {len(memories)} distinct memories. "
        
        # Add a sentence about the specific memories
        if len(memories) == 1:
            narrative += f"I remember {memories[0]['title']}, which reminds me of {', '.join(memories[0].get('keywords', [])[:3])}. "
        else:
            memory_titles = [m['title'] for m in memories[:2]]
            narrative += f"Experiences like {' and '.join(memory_titles)} stand out to me. "
        
        # Add a sentence connecting emotions/feelings
        emotional_words = ["evokes", "stirs", "awakens", "brings forth"]
        narrative += f"This place {random.choice(emotional_words)} feelings of {random.choice(connected_keywords)} and {random.choice(connected_keywords)} within me."
    else:
        narrative = f"Collective memories of {location} reveal {len(memories)} shared experiences. "
        
        # Add a sentence about common themes
        narrative += f"People often associate this place with {', '.join(unique_keywords[:3])}. "
        
        # Add a sentence about broader significance
        significance = ["cultural significance", "historical importance", "personal meaning", "emotional resonance"]
        narrative += f"These shared memories highlight the {random.choice(significance)} that {location} holds for many people."
    
    # Format keywords for response
    formatted_keywords = [
        {"text": kw, "type": "primary"} for kw in primary_keywords
    ]
    
    # Add connected keywords
    formatted_keywords.extend([
        {"text": kw, "type": "connected"} for kw in connected_keywords
    ])
    
    return {
        "text": narrative,
        "keywords": formatted_keywords
    }