/**
 * Utilities for highlighting text in the UI
 */

/**
 * Highlight a search term within a text string
 * @param {string} text - The original text
 * @param {string} searchTerm - The term to highlight
 * @param {string} highlightClass - CSS class to apply (default: "bg-yellow-200 text-yellow-800")
 * @returns {Array} - Array of React elements and strings
 */
export const highlightSearchTerm = (text, searchTerm, highlightClass = "bg-yellow-200 text-yellow-800") => {
    if (!text || !searchTerm || searchTerm.trim() === '') {
      return text;
    }
  
    // Create a regular expression with the search term, case-insensitive
    // Escape special regex characters in the search term
    const escapeRegExp = (string) => {
      return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    };
    
    const regex = new RegExp(`(${escapeRegExp(searchTerm)})`, 'gi');
    
    // Split the text by the regex
    const parts = text.split(regex);
    
    // Map the parts, highlighting matches
    return parts.map((part, index) => {
      // Check if this part matches the search term (case insensitive)
      if (part.toLowerCase() === searchTerm.toLowerCase()) {
        return {
          text: part,
          highlighted: true,
          highlightClass: highlightClass,
          key: index
        };
      }
      
      return {
        text: part,
        highlighted: false,
        key: index
      };
    });
  };
  
  /**
   * Highlight multiple search terms within a text string
   * @param {string} text - The original text
   * @param {Array} searchTerms - Array of terms to highlight
   * @param {string} highlightClass - CSS class to apply
   * @returns {Array} - Array of text segments with highlight information
   */
  export const highlightMultipleTerms = (text, searchTerms = [], highlightClass = "bg-yellow-200 text-yellow-800") => {
    if (!text || !searchTerms.length) {
      return [{ text, highlighted: false, key: 0 }];
    }
    
    // Sort search terms by length (longest first) to avoid highlighting parts of longer terms
    const sortedTerms = [...searchTerms].sort((a, b) => b.length - a.length);
    
    // Escape special regex characters in the search terms
    const escapeRegExp = (string) => {
      return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    };
    
    // Create a regex pattern that matches any of the search terms
    const pattern = new RegExp(
      `(${sortedTerms.map(term => escapeRegExp(term)).join('|')})`, 
      'gi'
    );
    
    // Split the text by the pattern
    const parts = text.split(pattern);
    
    // Map the parts, highlighting matches
    return parts.map((part, index) => {
      // Check if this part matches any search term (case insensitive)
      const isHighlighted = sortedTerms.some(
        term => part.toLowerCase() === term.toLowerCase()
      );
      
      return {
        text: part,
        highlighted: isHighlighted,
        highlightClass: isHighlighted ? highlightClass : null,
        key: index
      };
    });
  };
  
  /**
   * Highlight terms enclosed in special markers (like *term*)
   * @param {string} text - The original text with marked terms
   * @param {string} marker - Marker character (default: '*')
   * @param {string} highlightClass - CSS class to apply
   * @returns {Array} - Array of text segments with highlight information
   */
  export const highlightMarkedTerms = (text, marker = '*', highlightClass = "font-bold text-red-600") => {
    if (!text) {
      return [{ text: '', highlighted: false, key: 0 }];
    }
    
    // Create a regex pattern that matches text between markers
    const pattern = new RegExp(`(\\${marker}[^${marker}]+\\${marker})`, 'g');
    
    // Split the text by the pattern
    const parts = text.split(pattern);
    
    // Map the parts, highlighting marked sections
    return parts.map((part, index) => {
      if (part.startsWith(marker) && part.endsWith(marker)) {
        // Remove markers and return highlighted text
        const content = part.substring(1, part.length - 1);
        return {
          text: content,
          highlighted: true,
          highlightClass: highlightClass,
          key: index
        };
      }
      
      return {
        text: part,
        highlighted: false,
        key: index
      };
    });
  };
  
  /**
   * Render highlighted text segments as React elements
   * @param {Array} segments - Array of text segments with highlight information
   * @returns {Array} - Array of React elements
   */
  export const renderHighlightedText = (segments) => {
    return segments.map(segment => {
      if (segment.highlighted) {
        return (
          <span 
            key={segment.key} 
            className={segment.highlightClass}
          >
            {segment.text}
          </span>
        );
      }
      return <span key={segment.key}>{segment.text}</span>;
    });
  };
  
  /**
   * Extract all highlighted terms from a text
   * @param {string} text - Text with marked terms
   * @param {string} marker - Marker character (default: '*')
   * @returns {Array} - Array of highlighted terms
   */
  export const extractHighlightedTerms = (text, marker = '*') => {
    if (!text) return [];
    
    const regex = new RegExp(`\\${marker}([^${marker}]+)\\${marker}`, 'g');
    const matches = text.match(regex) || [];
    
    // Remove markers from matches
    return matches.map(match => match.substring(1, match.length - 1));
  };