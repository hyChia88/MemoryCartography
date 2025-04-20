from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

class MemoryRecommendationEngine:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    
    def generate_synthetic_narrative(self, memories):
        """Generate a simple synthetic narrative from memories."""
        if not memories:
            return "No memories found to generate a narrative."
        
        # Sort memories by weight
        sorted_memories = sorted(memories, key=lambda m: m.get('weight', 1.0), reverse=True)
        
        # Select top memories
        selected_memories = sorted_memories[:3]
        
        # Create narrative text
        narrative_parts = []
        for memory in selected_memories:
            narrative_parts.append(
                f"On {memory.get('date', 'an unknown date')}, "
                f"I was in {memory.get('location', 'an unknown place')}. "
                f"{memory.get('description', '')}"
            )
        
        return " ".join(narrative_parts)
    
    def find_similar_memories(self, source_memory, all_memories, top_n=5):
        """Find similar memories using TF-IDF and cosine similarity."""
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
        
        # Get top similar memories (excluding the source memory itself)
        similar_indices = similarities.argsort()[::-1][1:top_n+1]
        
        return [all_memories[idx] for idx in similar_indices]