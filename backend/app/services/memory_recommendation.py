# memory_recommendation.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
from datetime import datetime

class MemoryRecommendationEngine:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    
    # In memory_recommendation.py, add vector-based similarity:
    def find_similar_memories_by_embedding(self, source_embedding, all_memories, top_n=5):
        """Find similar memories using pre-computed embeddings."""
        if not all_memories:
            return []
            
        # Extract embeddings from memories
        memory_embeddings = []
        for memory in all_memories:
            try:
                embedding = json.loads(memory.get('embedding', '[]'))
                memory_embeddings.append(embedding)
            except:
                # Use a zero vector as fallback
                memory_embeddings.append([0.0] * len(source_embedding))
                
        # Calculate cosine similarities
        similarities = cosine_similarity([source_embedding], memory_embeddings)[0]
        
        # Get top similar memories
        similar_indices = similarities.argsort()[::-1][:top_n]
        
        return [all_memories[idx] for idx in similar_indices]

    def find_similar_memories(self, source_memory, all_memories, top_n=5):
        """Find similar memories using TF-IDF and cosine similarity."""
        if not all_memories:
            return []
            
        # Prepare text for vectorization
        texts = [
            f"{m.get('title', '')} {m.get('description', '')} {' '.join(m.get('keywords', []))}" 
            for m in all_memories
        ]
        
        # Vectorize texts
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        
        # Vectorize source memory
        source_text = (
            f"{source_memory.get('title', '')} "
            f"{source_memory.get('description', '')} "
            f"{' '.join(source_memory.get('keywords', []))}"
        )
        source_vector = self.tfidf_vectorizer.transform([source_text])
        
        # Calculate cosine similarities
        similarities = cosine_similarity(source_vector, tfidf_matrix)[0]
        
        # Get top similar memories (excluding the source memory itself if it's in all_memories)
        sorted_indices = similarities.argsort()[::-1]
        
        # Filter out the source memory if it's in the list
        similar_indices = [
            idx for idx in sorted_indices 
            if all_memories[idx].get('id') != source_memory.get('id')
        ][:top_n]
        
        return [all_memories[idx] for idx in similar_indices]
    
    def sort_memories_by_relevance(self, memories, target_weight=5.0):
        """
        Sort memories by relevance and select ones until their total weight 
        reaches or exceeds the target weight.
        """
        if not memories:
            return []
            
        # Sort by weight in descending order
        sorted_memories = sorted(memories, key=lambda m: m.get('weight', 1.0), reverse=True)
        
        # Select memories until total weight reaches target
        selected_memories = []
        total_weight = 0.0
        
        for memory in sorted_memories:
            memory_weight = memory.get('weight', 1.0)
            selected_memories.append(memory)
            total_weight += memory_weight
            
            if total_weight >= target_weight:
                break
                
        return selected_memories
    
    def sort_memories_by_recency(self, memories, max_memories=30):
        """Sort memories by date (most recent first)."""
        if not memories:
            return []
            
        # Sort by date in descending order (most recent first)
        sorted_memories = sorted(
            memories, 
            key=lambda m: datetime.strptime(m.get('date', '1900-01-01'), '%Y-%m-%d'),
            reverse=True
        )
        
        return sorted_memories[:max_memories]