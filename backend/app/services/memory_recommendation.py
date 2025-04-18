# backend/app/services/memory_recommendation.py
import sqlite3
import json
import random
import numpy as np
from collections import Counter
from pathlib import Path

class MemoryRecommendationEngine:
    def __init__(self, db_path='data/metadata/memories.db', 
                 user_images_dir='data/processed/user_photos',
                 public_images_dir='data/processed/public_photos'):
        self.db_path = db_path
        self.user_images_dir = Path(user_images_dir)
        self.public_images_dir = Path(public_images_dir)
    
    def _get_connection(self):
        """Get a database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def get_memories_by_location(self, location, memory_type="user", limit=5):
        """Retrieve memories by location."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Find memories with matching location
        cursor.execute(
            """
            SELECT id, filename, title, location, date, keywords, description
            FROM memories 
            WHERE type = ? AND location LIKE ?
            LIMIT ?
            """,
            (memory_type, f"%{location}%", limit)
        )
        memories = cursor.fetchall()
        
        # If no direct matches, try to find by keywords
        if not memories:
            cursor.execute(
                """
                SELECT id, filename, title, location, date, keywords, description
                FROM memories 
                WHERE type = ?
                """,
                (memory_type,)
            )
            all_memories = cursor.fetchall()
            
            # Find memories with location in keywords
            matching_memories = []
            for memory in all_memories:
                if not memory['keywords']:
                    continue
                
                keywords = json.loads(memory['keywords'])
                if any(location.lower() in keyword.lower() for keyword in keywords):
                    matching_memories.append(dict(memory))
            
            # Sort by relevance (how many keywords match)
            matching_memories.sort(
                key=lambda m: sum(1 for k in json.loads(m['keywords']) if location.lower() in k.lower()),
                reverse=True
            )
            
            memories = matching_memories[:limit]
        else:
            # Convert to list of dicts
            memories = [dict(memory) for memory in memories]
        
        # Add image paths
        for memory in memories:
            memory['keywords'] = json.loads(memory['keywords']) if memory['keywords'] else []
            if memory_type == 'user':
                memory['image_path'] = str(self.user_images_dir / memory['filename'])
            else:
                memory['image_path'] = str(self.public_images_dir / memory['filename'])
        
        conn.close()
        return memories
    
    def get_related_memories(self, keyword, memory_type="user", exclude_ids=None, limit=3):
        """Find memories related by keyword."""
        if exclude_ids is None:
            exclude_ids = []
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Find memories containing the keyword
        placeholders = ','.join('?' for _ in exclude_ids)
        exclude_clause = f"AND id NOT IN ({placeholders})" if exclude_ids else ""
        
        query = f"""
            SELECT id, filename, title, location, date, keywords, description
            FROM memories 
            WHERE type = ? AND keywords LIKE ? {exclude_clause}
            LIMIT ?
        """
        
        params = [memory_type, f"%{keyword}%"] + exclude_ids + [limit]
        
        cursor.execute(query, params)
        memories = [dict(memory) for memory in cursor.fetchall()]
        
        # Add image paths and parse keywords
        for memory in memories:
            memory['keywords'] = json.loads(memory['keywords']) if memory['keywords'] else []
            if memory_type == 'user':
                memory['image_path'] = str(self.user_images_dir / memory['filename'])
            else:
                memory['image_path'] = str(self.public_images_dir / memory['filename'])
        
        conn.close()
        return memories
    
    def generate_memory_narrative(self, location, memory_type="user"):
        """Generate a narrative about a location based on memories."""
        # Get primary memories for this location
        primary_memories = self.get_memories_by_location(location, memory_type)
        
        if not primary_memories:
            return {
                "text": f"No memories found for {location}.",
                "keywords": [{"text": location, "type": "primary"}],
                "memories": []
            }
        
        # Extract all keywords from primary memories
        all_keywords = []
        for memory in primary_memories:
            all_keywords.extend(memory['keywords'])
        
        # Count keyword frequencies and get most common
        keyword_counts = Counter(all_keywords)
        most_common = [k for k, _ in keyword_counts.most_common(5)]
        
        # For each common keyword, find related memories
        exclude_ids = [memory['id'] for memory in primary_memories]
        connected_memories = []
        
        for keyword in most_common:
            related = self.get_related_memories(
                keyword, 
                memory_type, 
                exclude_ids=exclude_ids,
                limit=2
            )
            for memory in related:
                memory['connection_keyword'] = keyword
                connected_memories.append(memory)
        
        # Limit connected memories to avoid overwhelming
        if len(connected_memories) > 5:
            connected_memories = random.sample(connected_memories, 5)
        
        # Generate primary keywords (location + common keywords)
        primary_keywords = [{"text": location, "type": "primary"}]
        primary_keywords.extend([{"text": kw, "type": "primary"} for kw in most_common])
        
        # Generate connected keywords (from related memories)
        connected_locations = list(set(m['location'] for m in connected_memories))
        connected_keywords = [{"text": loc.split(',')[0], "type": "connected"} 
                             for loc in connected_locations[:3]]
        
        # Emotional concepts that might be associated
        emotional_concepts = ["nostalgia", "memory", "reflection", "experience", "journey"]
        connected_keywords.extend([
            {"text": concept, "type": "connected"} 
            for concept in random.sample(emotional_concepts, min(2, len(emotional_concepts)))
        ])
        
        # Generate narrative text
        if memory_type == "user":
            narrative = self._generate_user_narrative(location, primary_memories, connected_memories)
        else:
            narrative = self._generate_public_narrative(location, primary_memories, connected_memories)
        
        # Combine all data
        result = {
            "text": narrative,
            "keywords": primary_keywords + connected_keywords,
            "primary_memories": primary_memories,
            "connected_memories": connected_memories
        }
        
        return result
    
    def _generate_user_narrative(self, location, primary_memories, connected_memories):
        """Generate a personal narrative about user memories."""
        # First paragraph: About primary memories
        main_text = f"When I think of {location}, {len(primary_memories)} memories come to mind. "
        
        if primary_memories:
            dates = [m['date'] for m in primary_memories]
            dates.sort()
            earliest = dates[0]
            latest = dates[-1]
            
            if earliest != latest:
                main_text += f"My experiences there span from {earliest} to {latest}. "
            else:
                main_text += f"I remember being there on {earliest}. "
            
            # Describe the location based on keywords
            all_keywords = []
            for memory in primary_memories:
                all_keywords.extend(memory['keywords'])
            
            top_keywords = [k for k, _ in Counter(all_keywords).most_common(3)]
            if top_keywords:
                main_text += f"To me, this place brings to mind {', '.join(top_keywords)}. "
        
        # Second paragraph: About connected memories
        if connected_memories:
            connected_locations = list(set(m['location'] for m in connected_memories))
            connection_keywords = list(set(m['connection_keyword'] for m in connected_memories))
            
            main_text += f"\nInterestingly, these memories connect to others from {', '.join(connected_locations[:2])}. "
            main_text += f"The common threads that link these places in my mind are {', '.join(connection_keywords[:3])}. "
            main_text += "This illustrates how my memories stack and link across time and place, sometimes distorting the original events."
        
        return main_text
    
    def _generate_public_narrative(self, location, primary_memories, connected_memories):
        """Generate a collective narrative about public memories."""
        # First paragraph: About primary memories
        main_text = f"In collective memory, {location} is represented through {len(primary_memories)} shared experiences. "
        
        if primary_memories:
            # Describe the location based on keywords
            all_keywords = []
            for memory in primary_memories:
                all_keywords.extend(memory['keywords'])
            
            top_keywords = [k for k, _ in Counter(all_keywords).most_common(3)]
            if top_keywords:
                main_text += f"People commonly associate this place with {', '.join(top_keywords)}. "
            
            # Comment on cultural significance
            significance = ["cultural importance", "historical value", "shared experiences", "common perceptions"]
            main_text += f"These shared images showcase the {random.choice(significance)} of this location. "
        
        # Second paragraph: About connected memories
        if connected_memories:
            connected_locations = list(set(m['location'] for m in connected_memories))
            connection_keywords = list(set(m['connection_keyword'] for m in connected_memories))
            
            main_text += f"\nCollective memory also connects this place to {', '.join(connected_locations[:2])}, "
            main_text += f"through shared concepts like {', '.join(connection_keywords[:3])}. "
            main_text += "This demonstrates how group memories can form patterns that transcend individual experiences."
        
        return main_text

# Create a singleton instance
_recommendation_engine = None

def get_recommendation_engine():
    global _recommendation_engine
    if _recommendation_engine is None:
        _recommendation_engine = MemoryRecommendationEngine()
    return _recommendation_engine