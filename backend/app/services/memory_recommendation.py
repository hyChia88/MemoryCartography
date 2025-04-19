# backend/app/services/memory_recommendation.py
import sqlite3
import json
import random
import numpy as np
from collections import Counter
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

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
    
    def update_memory_weight(self, memory_id, new_weight=None, increment=0.1):
        """Update the weight of a memory to indicate it's more important."""
        if new_weight is None:
            # Get current weight and increment it
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT weight FROM memories WHERE id = ?", (memory_id,))
            result = cursor.fetchone()
            
            if not result:
                conn.close()
                return False
                
            current_weight = result['weight'] or 1.0
            new_weight = current_weight + increment
            
        # Update the weight
        conn = self._get_connection() if 'conn' not in locals() else conn
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE memories SET weight = ? WHERE id = ?",
            (new_weight, memory_id)
        )
        conn.commit()
        conn.close()
        return True
    
    def search_memories(self, search_term, memory_type="user", limit=10):
        """
        Search memories based on keywords, location, or date
        Returns memories ordered by relevance
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Split search term into keywords
        search_keywords = search_term.lower().split()
        
        # Get all memories of the specified type
        cursor.execute(
            """
            SELECT id, filename, title, location, date, keywords, description, weight, embedding
            FROM memories 
            WHERE type = ?
            """,
            (memory_type,)
        )
        all_memories = cursor.fetchall()
        
        # Calculate relevance score for each memory
        memory_scores = []
        for memory in all_memories:
            score = 0
            
            # Match in location (highest priority)
            if memory['location'] and any(kw.lower() in memory['location'].lower() for kw in search_keywords):
                score += 5.0
            
            # Match in title
            if memory['title'] and any(kw.lower() in memory['title'].lower() for kw in search_keywords):
                score += 3.0
            
            # Match in keywords
            if memory['keywords']:
                keywords = json.loads(memory['keywords'])
                for kw in search_keywords:
                    for memory_kw in keywords:
                        if kw.lower() in memory_kw.lower():
                            score += 2.0
                            break
            
            # Match in description
            if memory['description'] and any(kw.lower() in memory['description'].lower() for kw in search_keywords):
                score += 1.0
                
            # Adjust score by memory weight
            score *= float(memory['weight'] or 1.0)
            
            if score > 0:
                memory_dict = dict(memory)
                memory_dict['relevance_score'] = score
                memory_dict['keywords'] = json.loads(memory_dict['keywords']) if memory_dict['keywords'] else []
                memory_dict['embedding'] = json.loads(memory_dict['embedding']) if memory_dict['embedding'] else None
                
                # Calculate matching keywords
                matching_keywords = []
                for kw in search_keywords:
                    matches = [memory_kw for memory_kw in memory_dict['keywords'] 
                               if kw.lower() in memory_kw.lower()]
                    matching_keywords.extend(matches)
                
                memory_dict['matching_keywords'] = list(set(matching_keywords))
                memory_scores.append(memory_dict)
        
        # Sort by relevance score (descending)
        memory_scores.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Add image paths and return top results
        results = memory_scores[:limit]
        for memory in results:
            if memory_type == 'user':
                memory['image_path'] = str(self.user_images_dir / memory['filename'])
            else:
                memory['image_path'] = str(self.public_images_dir / memory['filename'])
        
        conn.close()
        return results
    
    def find_similar_memories(self, memory_id, memory_type="user", limit=5):
        """Find memories similar to a given memory based on embeddings and keywords."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Get the source memory
        cursor.execute(
            """
            SELECT id, filename, title, location, date, keywords, description, weight, embedding
            FROM memories 
            WHERE id = ?
            """,
            (memory_id,)
        )
        source_memory = cursor.fetchone()
        
        if not source_memory:
            conn.close()
            return []
        
        # Get all memories of the same type
        cursor.execute(
            """
            SELECT id, filename, title, location, date, keywords, description, weight, embedding
            FROM memories 
            WHERE type = ? AND id != ?
            """,
            (memory_type, memory_id)
        )
        all_memories = cursor.fetchall()
        
        # Calculate similarity scores
        similarity_scores = []
        
        source_embedding = json.loads(source_memory['embedding']) if source_memory['embedding'] else None
        source_keywords = json.loads(source_memory['keywords']) if source_memory['keywords'] else []
        
        for memory in all_memories:
            score = 0
            
            # Keyword similarity (count shared keywords)
            if memory['keywords'] and source_keywords:
                target_keywords = json.loads(memory['keywords'])
                shared_keywords = set(source_keywords).intersection(set(target_keywords))
                keyword_score = len(shared_keywords) * 0.5  # 0.5 points per shared keyword
                score += keyword_score
            
            # Location similarity
            if source_memory['location'] == memory['location']:
                score += 3.0
            elif source_memory['location'] in memory['location'] or memory['location'] in source_memory['location']:
                score += 1.5
            
            # Embedding similarity (if both have embeddings)
            memory_embedding = json.loads(memory['embedding']) if memory['embedding'] else None
            if source_embedding and memory_embedding:
                try:
                    embedding_similarity = cosine_similarity(
                        [source_embedding], 
                        [memory_embedding]
                    )[0][0]
                    score += embedding_similarity * 5.0  # Scale up embedding similarity
                except:
                    pass
            
            # Apply memory weight
            score *= float(memory['weight'] or 1.0)
            
            if score > 0:
                memory_dict = dict(memory)
                memory_dict['similarity_score'] = score
                memory_dict['keywords'] = json.loads(memory_dict['keywords']) if memory_dict['keywords'] else []
                similarity_scores.append(memory_dict)
        
        # Sort by similarity score
        similarity_scores.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Add image paths and return top results
        results = similarity_scores[:limit]
        for memory in results:
            if memory_type == 'user':
                memory['image_path'] = str(self.user_images_dir / memory['filename'])
            else:
                memory['image_path'] = str(self.public_images_dir / memory['filename'])
        
        conn.close()
        return results
    
    def generate_synthetic_memory(self, memories, min_total_weight=3.0):
        """
        Generate synthetic memory based on a collection of memories.
        Prioritizes memories with higher weights until reaching min_total_weight.
        """
        if not memories:
            return {
                "text": "No memories available to generate a synthetic memory.",
                "keywords": []
            }
        
        # Sort memories by weight (descending)
        sorted_memories = sorted(memories, key=lambda m: float(m.get('weight', 1.0)), reverse=True)
        
        # Select memories until we reach the minimum total weight
        selected_memories = []
        total_weight = 0.0
        for memory in sorted_memories:
            selected_memories.append(memory)
            total_weight += float(memory.get('weight', 1.0))
            if total_weight >= min_total_weight:
                break
        
        # If we don't have enough memories, use all of them
        if not selected_memories:
            selected_memories = sorted_memories
        
        # Extract information from selected memories
        memory_dates = [m.get('date', '') for m in selected_memories]
        memory_titles = [m.get('title', '') for m in selected_memories]
        memory_locations = [m.get('location', '') for m in selected_memories]
        memory_descriptions = [m.get('description', '') for m in selected_memories]
        
        # Collect all keywords
        all_keywords = []
        for memory in selected_memories:
            keywords = memory.get('keywords', [])
            if keywords:
                all_keywords.extend(keywords)
        
        # Count occurrences of each keyword
        keyword_counts = Counter(all_keywords)
        
        # Get the most common keywords
        common_keywords = [k for k, _ in keyword_counts.most_common(10)]
        
        # Generate a synthetic memory description
        synthetic_text = self._generate_synthetic_text(
            memory_dates, memory_titles, memory_locations, memory_descriptions, common_keywords
        )
        
        return {
            "text": synthetic_text,
            "keywords": common_keywords,
            "source_memories": selected_memories
        }
    
    def _generate_synthetic_text(self, dates, titles, locations, descriptions, keywords):
        """Generate synthetic text based on memory information."""
        # Sort dates for chronological ordering
        sorted_indices = sorted(range(len(dates)), key=lambda i: dates[i] if dates[i] else "")
        
        sorted_dates = [dates[i] for i in sorted_indices]
        sorted_titles = [titles[i] for i in sorted_indices]
        sorted_locations = [locations[i] for i in sorted_indices]
        sorted_descriptions = [descriptions[i] for i in sorted_indices]
        
        # Initialize synthetic text
        synthetic_text = ""
        
        # Add introduction based on time span
        if sorted_dates:
            earliest_date = sorted_dates[0]
            latest_date = sorted_dates[-1]
            
            if earliest_date == latest_date:
                synthetic_text += f"On {earliest_date}, "
            else:
                synthetic_text += f"From {earliest_date} to {latest_date}, "
            
            # Add locations
            unique_locations = list(set(sorted_locations))
            if len(unique_locations) == 1:
                synthetic_text += f"I spent time in {unique_locations[0]}. "
            else:
                synthetic_text += f"I traveled between {', '.join(unique_locations[:3])}. "
        
        # Add content from descriptions
        if sorted_descriptions:
            # Use most important keywords to guide the summary
            if keywords:
                top_keywords = keywords[:5]
                synthetic_text += f"These experiences were characterized by {', '.join(top_keywords)}. "
            
            # Add brief summary of experiences
            synthetic_text += "I remember "
            
            if len(sorted_descriptions) == 1:
                # Paraphrase the single description
                words = sorted_descriptions[0].split()
                if len(words) > 10:
                    synthetic_text += " ".join(words[:10]) + "... "
                else:
                    synthetic_text += sorted_descriptions[0] + " "
            else:
                # Mention multiple experiences
                synthetic_text += f"several experiences including {sorted_titles[0]}"
                if len(sorted_titles) > 1:
                    synthetic_text += f" and {sorted_titles[1]}"
                synthetic_text += ". "
            
            # Add emotional reflection
            emotions = ["joy", "nostalgia", "awe", "excitement", "peace", "wonder"]
            synthetic_text += f"These memories evoke feelings of {random.choice(emotions)} and {random.choice(emotions)}."
        
        return synthetic_text
        
    def generate_memory_narrative(self, search_term, memory_type="user", min_weight=3.0, max_memories=10):
        """
        Generate a complete memory narrative based on a search term.
        Includes finding relevant memories and generating synthetic memory.
        """
        # Step 1: Find relevant memories based on search term
        relevant_memories = self.search_memories(
            search_term, 
            memory_type=memory_type,
            limit=max_memories
        )
        
        # Step 2: Generate synthetic memory from the relevant memories
        synthetic_memory = self.generate_synthetic_memory(
            relevant_memories,
            min_total_weight=min_weight
        )
        
        # Step 3: Find connected memories (memories similar to the relevant ones)
        connected_memories = []
        for memory in relevant_memories[:3]:  # Only use top 3 memories to find connections
            similar_memories = self.find_similar_memories(
                memory['id'],
                memory_type=memory_type,
                limit=3
            )
            connected_memories.extend(similar_memories)
        
        # Remove duplicates
        seen_ids = set()
        unique_connected_memories = []
        for memory in connected_memories:
            if memory['id'] not in seen_ids and memory['id'] not in [m['id'] for m in relevant_memories]:
                seen_ids.add(memory['id'])
                unique_connected_memories.append(memory)
        
        # Step 4: Format keywords for the response
        primary_keywords = []
        for kw in synthetic_memory['keywords'][:5]:  # Top 5 keywords
            primary_keywords.append({"text": kw, "type": "primary"})
            
        connected_keywords = []
        for memory in unique_connected_memories[:3]:  # Take keywords from top 3 connected memories
            if 'keywords' in memory and memory['keywords']:
                for kw in memory['keywords'][:2]:  # Just 2 keywords per connected memory
                    if kw not in [k["text"] for k in primary_keywords] and kw not in [k["text"] for k in connected_keywords]:
                        connected_keywords.append({"text": kw, "type": "connected"})
        
        # Create the final response
        response = {
            "text": synthetic_memory["text"],
            "keywords": primary_keywords + connected_keywords[:5],  # Limit to 5 connected keywords
            "primary_memories": relevant_memories,
            "connected_memories": unique_connected_memories[:5]  # Limit to 5 connected memories
        }
        
        return response