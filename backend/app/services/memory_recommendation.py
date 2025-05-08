# memory_recommendation.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
from datetime import datetime
import os

class MemoryRecommendationEngine:
    """Engine for finding and recommending memories based on similarity."""
    
    def __init__(self):
        """Initialize the recommendation engine with TF-IDF vectorizer."""
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    
    def search_memories_by_embeddings(self, query, all_memories, limit=30):
        """
        Search memories using TF-IDF text embeddings and combined scoring.
        
        Args:
            query (str): Search query text
            all_memories (list): List of memory dictionaries
            limit (int): Maximum number of results to return
            
        Returns:
            list: Sorted list of memories with similarity scores
        """
        if not all_memories:
            return []
        
        # Prepare search query terms
        query_terms = query.lower().split()
        
        # Step 1: Score memories based on text matching
        for memory in all_memories:
            score = 0
            
            # Check title
            if any(term in memory.get('title', '').lower() for term in query_terms):
                score += 5.0
            
            # Check location
            if any(term in memory.get('location', '').lower() for term in query_terms):
                score += 3.0
            
            # Check description
            if any(term in memory.get('description', '').lower() for term in query_terms):
                score += 2.0
            
            # Check keywords
            keywords = memory.get('keywords', [])
            if keywords and any(query_term in keyword.lower() for query_term in query_terms for keyword in keywords):
                score += 4.0
            
            # Check detected objects if available
            detected_objects = []
            if 'detected_objects' in memory and memory['detected_objects']:
                try:
                    detected_objects = json.loads(memory['detected_objects'])
                    if any(query_term in obj.lower() for query_term in query_terms for obj in detected_objects):
                        score += 3.0
                except json.JSONDecodeError:
                    pass
            
            # Store the initial text match score
            memory['text_match_score'] = score
        
        # Step 2: Generate TF-IDF embeddings for semantic similarity
        all_memory_texts = []
        for memory in all_memories:
            text_content = (
                f"{memory.get('title', '')} "
                f"{memory.get('location', '')} "
                f"{memory.get('description', '')} "
                f"{' '.join(memory.get('keywords', []))}"
            )
            all_memory_texts.append(text_content)
        
        # Add the query to ensure it's in the vocabulary
        all_texts = all_memory_texts + [query]
        
        # Create TF-IDF matrix
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)
            
            # Calculate cosine similarity between query and all memories
            query_vector = tfidf_matrix[-1]  # Last row is the query
            memory_vectors = tfidf_matrix[:-1]  # All other rows are memories
            
            similarities = cosine_similarity(query_vector, memory_vectors)[0]
            
            # Store similarity scores
            for i, memory in enumerate(all_memories):
                memory['semantic_similarity'] = float(similarities[i])
        except Exception as e:
            print(f"Error calculating semantic similarity: {e}")
            # Fallback: just use text match scores
            for memory in all_memories:
                memory['semantic_similarity'] = 0.0
        
        # Step 3: Calculate combined scores
        for memory in all_memories:
            # Calculate date score (normalized to 0-1 range)
            try:
                date_obj = datetime.strptime(memory.get('date', '1900-01-01'), '%Y-%m-%d')
                # Convert to timestamp and normalize (assuming dates within last 10 years)
                max_date = datetime.now()
                min_date = max_date.replace(year=max_date.year - 10)
                date_score = (date_obj.timestamp() - min_date.timestamp()) / (max_date.timestamp() - min_date.timestamp())
                date_score = min(1.0, max(0.0, date_score))  # Ensure it's between 0-1
            except:
                date_score = 0.0
            
            # Get weight score (normalized to 0-1 range)
            weight_score = float(memory.get('weight', 1.0)) / 5.0  # Assuming max weight is 5.0
            weight_score = min(1.0, max(0.0, weight_score))  # Ensure it's between 0-1
            
            # Get text match score (normalized)
            text_match_score = memory.get('text_match_score', 0.0) / 10.0  # Normalize by max possible score
            text_match_score = min(1.0, max(0.0, text_match_score))  # Ensure it's between 0-1
            
            # Get semantic similarity score (already between 0-1)
            semantic_score = memory.get('semantic_similarity', 0.0)
            
            # Calculate combined score with weights
            # 30% weight, 20% date, 20% text match, 30% semantic similarity
            memory['combined_score'] = (
                (weight_score * 0.15) + 
                (date_score * 0.15) + 
                (text_match_score * 0.3) + 
                (semantic_score * 0.4)
            )
        
        # Step 4: Sort by combined score and return top results
        sorted_memories = sorted(
            all_memories, 
            key=lambda m: m.get('combined_score', 0.0),
            reverse=True
        )
        
        # Remove temporary scoring fields for cleaner results
        result_memories = []
        for memory in sorted_memories[:limit]:
            memory_copy = memory.copy()
            for field in ['text_match_score', 'semantic_similarity', 'combined_score']:
                if field in memory_copy:
                    memory_copy.pop(field)
            result_memories.append(memory_copy)
        
        return result_memories
    
    def find_similar_memories(self, source_memory, all_memories, top_n=5):
        """
        Find memories similar to a source memory using TF-IDF and cosine similarity.
        
        Args:
            source_memory (dict): Source memory to find similar ones to
            all_memories (list): List of all memories to search within
            top_n (int): Number of top similar memories to return
            
        Returns:
            list: Top similar memories
        """
        if not all_memories:
            return []
        
        # Prepare text for vectorization
        texts = [
            f"{m.get('title', '')} {m.get('location', '')} {m.get('description', '')} {' '.join(m.get('keywords', []))}" 
            for m in all_memories
        ]
        
        # Vectorize texts
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        
        # Vectorize source memory
        source_text = (
            f"{source_memory.get('title', '')} "
            f"{source_memory.get('location', '')} "
            f"{source_memory.get('description', '')} "
            f"{' '.join(source_memory.get('keywords', []))}"
        )
        source_vector = self.tfidf_vectorizer.transform([source_text])
        
        # Calculate cosine similarities
        similarities = cosine_similarity(source_vector, tfidf_matrix)[0]
        
        # Get top similar memories (excluding the source memory itself)
        sorted_indices = similarities.argsort()[::-1]
        
        # Filter out the source memory if it's in the list
        similar_indices = [
            idx for idx in sorted_indices 
            if all_memories[idx].get('id') != source_memory.get('id')
        ][:top_n]
        
        return [all_memories[idx] for idx in similar_indices]
        
    def detect_objects_in_image(self, image_path, conf_threshold=0.25):
        """
        Placeholder for YOLO object detection.
        In a full implementation, this would detect objects in an image.
        
        Returns empty results as a simplified implementation.
        """
        print(f"Object detection would process: {image_path}")
        return None, None, []
        
    def extract_image_with_objects(self, image_path, output_path=None):
        """
        Placeholder for combined feature extraction and object detection.
        
        Returns empty results as a simplified implementation.
        """
        return {
            "features": None,
            "objects_detected": [],
            "annotated_image_base64": None,
            "success": False
        }
        
    def describe_visual_content(self, image_path):
        """
        Placeholder for image content description.
        
        Returns empty description as a simplified implementation.
        """
        return {
            "description": "Image content description would be generated here.",
            "objects": []
        }