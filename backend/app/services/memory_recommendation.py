# app/services/memory_recommendation.py
import logging
import json
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional

# Try to import scikit-learn for feature extraction and similarity
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available. Falling back to basic text matching.")

class MemoryRecommendationEngine:
    """Engine for finding and recommending memories based on similarity."""
    
    def __init__(self):
        """Initialize the recommendation engine."""
        # Initialize TF-IDF vectorizer if scikit-learn is available
        if SKLEARN_AVAILABLE:
            self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
            logging.info("Memory recommendation engine initialized with TF-IDF capabilities.")
        else:
            self.tfidf_vectorizer = None
            logging.info("Memory recommendation engine initialized with basic matching only.")
    
    def search_memories_by_embeddings(self, query: str, all_memories: List[Dict[str, Any]], limit: int = 30) -> List[Dict[str, Any]]:
        """
        Search memories using TF-IDF text embeddings and combined scoring.
        
        Args:
            query: Search query text
            all_memories: List of memory dictionaries
            limit: Maximum number of results to return
            
        Returns:
            Sorted list of memories with similarity scores
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
            
            # Check keywords (stored as JSON string in database)
            keywords = []
            if isinstance(memory.get('keywords'), str):
                try:
                    keywords = json.loads(memory.get('keywords', '[]'))
                except json.JSONDecodeError:
                    keywords = []
            elif isinstance(memory.get('keywords'), list):
                keywords = memory.get('keywords', [])
                
            if keywords and any(query_term in keyword.lower() for query_term in query_terms for keyword in keywords):
                score += 4.0
            
            # Check detected objects
            detected_objects = []
            if 'detected_objects' in memory:
                if isinstance(memory['detected_objects'], str):
                    try:
                        detected_objects = json.loads(memory['detected_objects'])
                    except json.JSONDecodeError:
                        detected_objects = []
                elif isinstance(memory['detected_objects'], list):
                    detected_objects = memory['detected_objects']
                    
                if any(query_term in obj.lower() for query_term in query_terms for obj in detected_objects):
                    score += 3.0
            
            # Store the initial text match score
            memory['text_match_score'] = score
        
        # Step 2: Generate TF-IDF embeddings for semantic similarity (if available)
        if SKLEARN_AVAILABLE and self.tfidf_vectorizer:
            try:
                # Prepare text content for TF-IDF
                all_memory_texts = []
                for memory in all_memories:
                    # Combine all textual information
                    text_content = (
                        f"{memory.get('title', '')} "
                        f"{memory.get('location', '')} "
                        f"{memory.get('description', '')} "
                    )
                    
                    # Add keywords if available
                    keywords = []
                    if isinstance(memory.get('keywords'), str):
                        try:
                            keywords = json.loads(memory.get('keywords', '[]'))
                        except json.JSONDecodeError:
                            keywords = []
                    elif isinstance(memory.get('keywords'), list):
                        keywords = memory.get('keywords', [])
                        
                    text_content += ' '.join(keywords)
                    
                    all_memory_texts.append(text_content)
                
                # Add the query to ensure it's in the vocabulary
                all_texts = all_memory_texts + [query]
                
                # Create TF-IDF matrix
                tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)
                
                # Calculate cosine similarity between query and all memories
                query_vector = tfidf_matrix[-1]  # Last row is the query
                memory_vectors = tfidf_matrix[:-1]  # All other rows are memories
                
                similarities = cosine_similarity(query_vector, memory_vectors)[0]
                
                # Store similarity scores
                for i, memory in enumerate(all_memories):
                    memory['semantic_similarity'] = float(similarities[i])
            except Exception as e:
                logging.error(f"Error calculating semantic similarity: {e}")
                # Fallback: just use text match scores
                for memory in all_memories:
                    memory['semantic_similarity'] = 0.0
        else:
            # Fallback if scikit-learn not available
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
            # 15% weight, 15% date, 30% text match, 40% semantic similarity
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
    
    def find_similar_memories(self, 
                             source_memory: Dict[str, Any], 
                             all_memories: List[Dict[str, Any]], 
                             top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Find memories similar to a source memory using TF-IDF and cosine similarity.
        
        Args:
            source_memory: Source memory to find similar ones to
            all_memories: List of all memories to search within
            top_n: Number of top similar memories to return
            
        Returns:
            Top similar memories
        """
        if not all_memories:
            return []
        
        # Basic similarity if scikit-learn not available
        if not SKLEARN_AVAILABLE or not self.tfidf_vectorizer:
            # Calculate basic similarity based on location and objects
            similarities = []
            source_location = source_memory.get('location', '').lower()
            
            # Get source detected objects
            source_objects = []
            if 'detected_objects' in source_memory:
                if isinstance(source_memory['detected_objects'], str):
                    try:
                        source_objects = json.loads(source_memory['detected_objects'])
                    except json.JSONDecodeError:
                        source_objects = []
                elif isinstance(source_memory['detected_objects'], list):
                    source_objects = source_memory['detected_objects']
            
            for i, memory in enumerate(all_memories):
                # Skip the source memory itself
                if memory.get('id') == source_memory.get('id'):
                    similarities.append(0)
                    continue
                
                score = 0
                
                # Location match
                if source_location and source_location in memory.get('location', '').lower():
                    score += 0.5
                
                # Object overlap
                memory_objects = []
                if 'detected_objects' in memory:
                    if isinstance(memory['detected_objects'], str):
                        try:
                            memory_objects = json.loads(memory['detected_objects'])
                        except json.JSONDecodeError:
                            memory_objects = []
                    elif isinstance(memory['detected_objects'], list):
                        memory_objects = memory['detected_objects']
                
                if source_objects and memory_objects:
                    overlap = set(source_objects).intersection(set(memory_objects))
                    score += len(overlap) * 0.2
                
                similarities.append(score)
            
            # Get indices of top similar memories
            sorted_indices = np.argsort(similarities)[::-1]
            
            # Filter out the source memory if it's in the list
            similar_indices = [
                idx for idx in sorted_indices 
                if all_memories[idx].get('id') != source_memory.get('id')
            ][:top_n]
            
            return [all_memories[idx] for idx in similar_indices]
        
        # TF-IDF based similarity if scikit-learn is available
        try:
            # Prepare text for vectorization
            texts = []
            for m in all_memories:
                # Combine all textual information
                text_content = (
                    f"{m.get('title', '')} {m.get('location', '')} {m.get('description', '')} "
                )
                
                # Add keywords if available
                keywords = []
                if isinstance(m.get('keywords'), str):
                    try:
                        keywords = json.loads(m.get('keywords', '[]'))
                    except json.JSONDecodeError:
                        keywords = []
                elif isinstance(m.get('keywords'), list):
                    keywords = m.get('keywords', [])
                    
                text_content += ' '.join(keywords)
                
                texts.append(text_content)
            
            # Vectorize texts
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            
            # Vectorize source memory
            source_text = (
                f"{source_memory.get('title', '')} "
                f"{source_memory.get('location', '')} "
                f"{source_memory.get('description', '')} "
            )
            
            # Add keywords if available
            source_keywords = []
            if isinstance(source_memory.get('keywords'), str):
                try:
                    source_keywords = json.loads(source_memory.get('keywords', '[]'))
                except json.JSONDecodeError:
                    source_keywords = []
            elif isinstance(source_memory.get('keywords'), list):
                source_keywords = source_memory.get('keywords', [])
                
            source_text += ' '.join(source_keywords)
            
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
            
        except Exception as e:
            logging.error(f"Error finding similar memories: {e}")
            return []