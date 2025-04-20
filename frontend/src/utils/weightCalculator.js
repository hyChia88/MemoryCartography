// Utility for calculating and managing memory event weights
class WeightCalculator {
    // Calculate base weight for a memory event
    calculateBaseWeight(memoryEvent) {
      const {
        location = '',
        keywords = [],
        date = new Date().toISOString()
      } = memoryEvent;
  
      // Base calculations
      const locationWeight = this.calculateLocationWeight(location);
      const keywordWeight = this.calculateKeywordWeight(keywords);
      const dateWeight = this.calculateDateWeight(date);
  
      // Combine weights
      let totalWeight = (locationWeight + keywordWeight + dateWeight) / 3;
  
      // Ensure weight is between 0 and 10
      return Math.min(Math.max(totalWeight, 0), 10);
    }
  
    // Calculate weight based on location specificity
    calculateLocationWeight(location) {
      // More specific locations get higher weight
      const locationSpecificityMap = {
        'City Center': 1.0,
        'Metropolitan Area': 0.8,
        'Country': 0.5,
        'Continent': 0.3,
        'Unknown': 0.1
      };
  
      // Determine location specificity
      if (location.includes('City Center')) return locationSpecificityMap['City Center'];
      if (location.includes('Metropolitan')) return locationSpecificityMap['Metropolitan Area'];
      if (location.includes('Country')) return locationSpecificityMap['Country'];
      if (location.includes('Continent')) return locationSpecificityMap['Continent'];
      
      return locationSpecificityMap['Unknown'];
    }
  
    // Calculate weight based on keywords
    calculateKeywordWeight(keywords) {
      if (!keywords || keywords.length === 0) return 0;
  
      // More diverse and specific keywords increase weight
      const keywordQualityMap = {
        'location': 0.3,
        'sensory': 0.2,
        'emotional': 0.2,
        'abstract': 0.1,
        'concrete': 0.2
      };
  
      // Categorize and weight keywords
      const categorizedKeywords = {
        location: keywords.filter(kw => kw.includes('location')),
        sensory: keywords.filter(kw => kw.includes('smell') || kw.includes('sound') || kw.includes('texture')),
        emotional: keywords.filter(kw => kw.includes('feeling') || kw.includes('mood')),
        abstract: keywords.filter(kw => kw.length > 10 && !/\d/.test(kw)),
        concrete: keywords.filter(kw => /\w+/.test(kw) && kw.length <= 10)
      };
  
      // Calculate keyword weight
      const keywordWeight = Object.entries(categorizedKeywords).reduce(
        (total, [category, matchedKeywords]) => 
          total + (matchedKeywords.length * keywordQualityMap[category]), 
        0
      );
  
      return Math.min(keywordWeight, 1.0);
    }
  
    // Calculate weight based on recency
    calculateDateWeight(dateString) {
      const date = new Date(dateString);
      const now = new Date();
      const daysDifference = (now - date) / (1000 * 60 * 60 * 24);
  
      // More recent memories get higher weight
      if (daysDifference <= 30) return 1.0;  // Within a month
      if (daysDifference <= 90) return 0.8;  // Within a quarter
      if (daysDifference <= 365) return 0.5;  // Within a year
      if (daysDifference <= 730) return 0.3;  // Within two years
      return 0.1;  // Older memories
    }
  
    // Increment weight when a memory is interacted with
    incrementWeight(currentWeight, incrementAmount = 0.5) {
      return Math.min(currentWeight + incrementAmount, 10);
    }
  
    // Decay weight over time for less relevant memories
    decayWeight(currentWeight, lastInteractionDate) {
      const daysSinceInteraction = (new Date() - new Date(lastInteractionDate)) / (1000 * 60 * 60 * 24);
      
      // Gradual weight decay
      const decayFactor = Math.max(0.1, 1 - (daysSinceInteraction / 365));
      
      return Math.max(0, currentWeight * decayFactor);
    }
  }
  
  export default new WeightCalculator();