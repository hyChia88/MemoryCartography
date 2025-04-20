// Utility for highlighting keywords in text
class HighlightUtils {
    // Highlight keywords in a text string
    highlightKeywords(text, keywords, options = {}) {
      const {
        highlightColor = 'red',
        highlightIntensity = 600
      } = options;
  
      if (!text || !keywords || keywords.length === 0) {
        return text;
      }
  
      // Create a regex pattern that matches any of the keywords
      const escapedKeywords = keywords.map(kw => 
        kw.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
      );
      const pattern = new RegExp(`\\b(${escapedKeywords.join('|')})\\b`, 'gi');
  
      // Split the text and wrap matched keywords
      const parts = text.split(pattern);
      
      return parts.map((part, index) => {
        const isKeyword = keywords.some(kw => 
          part.toLowerCase() === kw.toLowerCase()
        );
  
        if (isKeyword) {
          return (
            <span 
              key={index} 
              className={`font-bold text-${highlightColor}-${highlightIntensity}`}
            >
              {part}
            </span>
          );
        }
        
        return part;
      });
    }
  
    // Extract highlighted keywords from text
    extractHighlightedKeywords(text, keywords) {
      if (!text || !keywords) return [];
  
      const foundKeywords = keywords.filter(keyword => 
        new RegExp(`\\b${keyword}\\b`, 'gi').test(text)
      );
  
      return foundKeywords;
    }
  
    // Score text based on keyword coverage
    calculateKeywordCoverage(text, keywords) {
      if (!text || !keywords) return 0;
  
      const matchedKeywords = keywords.filter(keyword => 
        new RegExp(`\\b${keyword}\\b`, 'gi').test(text)
      );
  
      return matchedKeywords.length / keywords.length;
    }
  }
  
  export default new HighlightUtils();