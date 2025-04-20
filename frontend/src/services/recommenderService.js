class RecommenderService {
    // Calculate similarity score between keywords
    calculateKeywordSimilarity(searchTerms, memoryKeywords) {
      // Convert all terms to lowercase for case-insensitive matching
      const normalizedSearch = searchTerms.map(term => term.toLowerCase());
      const normalizedMemoryKeywords = memoryKeywords.map(kw => kw.toLowerCase());
  
      // Calculate matching keywords
      const matchedKeywords = normalizedSearch.filter(term => 
        normalizedMemoryKeywords.some(kw => kw.includes(term))
      );
  
      // Calculate score based on number of matched keywords and their coverage
      const matchScore = matchedKeywords.length / searchTerms.length;
      
      return matchScore;
    }
  
    // Recommend memories based on search terms
    recommendMemories(searchTerms, memories, limit = 10) {
      // Validate inputs
      if (!searchTerms || searchTerms.length === 0 || !memories) {
        return [];
      }
  
      // Calculate similarity scores for each memory
      const scoredMemories = memories.map(memory => ({
        ...memory,
        similarityScore: this.calculateKeywordSimilarity(searchTerms, memory.keywords)
      }));
  
      // Sort memories by similarity score in descending order
      const rankedMemories = scoredMemories
        .filter(memory => memory.similarityScore > 0)  // Only include memories with some similarity
        .sort((a, b) => b.similarityScore - a.similarityScore)
        .slice(0, limit);  // Limit results
  
      return rankedMemories;
    }
  
    // Increase weight of a memory when interacted with
    incrementMemoryWeight(memory, incrementAmount = 0.5) {
      return {
        ...memory,
        weight: Math.min(memory.weight + incrementAmount, 10)  // Cap weight at 10
      };
    }
  }
  
  export default new RecommenderService();