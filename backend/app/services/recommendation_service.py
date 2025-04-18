import json
import sqlite3
import numpy as np
import random
from collections import Counter

class RecommendationService:
    def __init__(self, db_path='data/metadata/memories.db'):
        self.db_path = db_path
    
    def _get_connection(self):
        """Get a database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def get_similar_memories(self, location, memory_type="user", limit=5):
        """Get memories similar to a location."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # First try direct location match
        cursor.execute(
            """
            SELECT id, title, location, date, keywords, content, embedding
            FROM memories 
            WHERE type = ? AND location LIKE ?
            """,
            (memory_type, f"%{location}%")
        )
        direct_matches = cursor.fetchall()
        
        # If no direct matches, find similar locations based on keywords
        if not direct_matches:
            # Get all memories
            cursor.execute(
                """
                SELECT id, title, location, date, keywords, content, embedding
                FROM memories 
                WHERE type = ?
                """,
                (memory_type,)
            )
            all_memories = cursor.fetchall()
            
            # Find memories with similar keywords
            similar_memories = []
            for memory in all_memories:
                if not memory['keywords']:
                    continue
                    
                keywords = json.loads(memory['keywords'])
                # Check if location is in keywords
                if any(location.lower() in kw.lower() for kw in keywords):
                    similar_memories.append(memory)
            
            # If we found any similar memories, use those
            if similar_memories:
                direct_matches = similar_memories
        
        conn.close()
        
        # Convert to dictionaries
        memories = []
        for memory in direct_matches[:limit]:
            memories.append({
                "id": memory['id'],
                "title": memory['title'],
                "location": memory['location'],
                "date": memory['date'],
                "keywords": json.loads(memory['keywords']) if memory['keywords'] else [],
                "content": memory['content'],
                "embedding": json.loads(memory['embedding']) if memory['embedding'] else None
            })
        
        return memories
    
    def get_semantic_connections(self, memories, all_type="user", limit=5):
        """Find semantic connections between memories."""
        if not memories:
            return []
        
        # Extract all keywords from the given memories
        all_keywords = []
        for memory in memories:
            all_keywords.extend(memory.get('keywords', []))
        
        # Count keyword frequencies
        keyword_counts = Counter(all_keywords)
        most_common = keyword_counts.most_common(5)
        
        # Get common keywords
        common_keywords = [k[0] for k in most_common]
        
        # Find memories that share these keywords
        conn = self._get_connection()
        cursor = conn.cursor()
        
        connected_memories = []
        for keyword in common_keywords:
            cursor.execute(
                """
                SELECT id, title, location, date, keywords, content
                FROM memories 
                WHERE type = ? AND keywords LIKE ?
                """,
                (all_type, f"%{keyword}%")
            )
            results = cursor.fetchall()
            
            for result in results:
                # Check if keyword is actually in the list (not just a substring)
                memory_keywords = json.loads(result['keywords']) if result['keywords'] else []
                if keyword in memory_keywords:
                    connected_memories.append({
                        "id": result['id'],
                        "title": result['title'],
                        "location": result['location'],
                        "date": result['date'],
                        "keywords": memory_keywords,
                        "content": result['content'],
                        "connection_type": "keyword",
                        "connection_value": keyword
                    })
        
        conn.close()
        
        # Remove duplicates (based on id) and limit results
        unique_memories = {}
        for memory in connected_memories:
            if memory['id'] not in unique_memories:
                unique_memories[memory['id']] = memory
        
        # Get a random sample of the connected memories (to add variability)
        connected_memory_list = list(unique_memories.values())
        
        if len(connected_memory_list) > limit:
            connected_memory_list = random.sample(connected_memory_list, limit)
        
        return connected_memory_list
    
    def enhance_narrative(self, narrative_data, memory_type="user"):
        """Enhance a narrative with semantic connections."""
        location = narrative_data.get('location', '')
        text = narrative_data.get('text', '')
        keywords = narrative_data.get('keywords', [])
        
        # Get memories related to this location
        related_memories = self.get_similar_memories(location, memory_type)
        
        # Find semantic connections to these memories
        connected_memories = self.get_semantic_connections(related_memories, memory_type)
        
        # Extract connected keywords
        connected_keywords = []
        for memory in connected_memories:
            connected_keyword = memory.get('connection_value')
            if connected_keyword:
                connected_keywords.append(connected_keyword)
        
        # Make connected keywords unique
        connected_keywords = list(set(connected_keywords))
        
        # Add connected keywords to existing keywords
        for kw in connected_keywords:
            if kw not in [k.get('text') for k in keywords]:
                keywords.append({"text": kw, "type": "connected"})
        
        # Enhanced text with connections
        if connected_memories and text:
            connected_locations = list(set(m.get('location') for m in connected_memories if m.get('location')))
            
            if connected_locations and memory_type == "user":
                text += f" These memories also connect to other places I've experienced like {', '.join(connected_locations[:3])}."
            elif connected_locations:
                text += f" These collective experiences share themes with other locations like {', '.join(connected_locations[:3])}."
        
        return {
            "text": text,
            "keywords": keywords
        }

# Singleton instance
_instance = None

def get_recommendation_service():
    global _instance
    if _instance is None:
        _instance = RecommendationService()
    return _instance