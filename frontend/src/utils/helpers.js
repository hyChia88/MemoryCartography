/**
 * Format a date string for display
 * 
 * @param {string} dateString - ISO date string
 * @returns {string} - Formatted date
 */
export const formatDate = (dateString) => {
    try {
      const date = new Date(dateString);
      return date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
      });
    } catch (e) {
      return dateString;
    }
  };
  
  /**
   * Sort memories by specified criteria
   * 
   * @param {Array} memories - Array of memories to sort
   * @param {string} sortOption - Sort option ('weight', 'date', etc)
   * @returns {Array} - Sorted memories
   */
  export const sortMemories = (memories, sortOption) => {
    if (!memories || !Array.isArray(memories) || memories.length === 0) {
      return [];
    }
  
    return [...memories].sort((a, b) => {
      if (sortOption === 'weight') {
        return (b.weight || 1.0) - (a.weight || 1.0);
      } else if (sortOption === 'date') {
        return new Date(b.date).getTime() - new Date(a.date).getTime();
      }
      // Default: don't sort, use order from API
      return 0;
    });
  };
  
  /**
   * Highlight keywords in text
   * 
   * @param {string} text - Text to highlight
   * @param {Array} keywords - Keywords to highlight
   * @returns {string} - HTML string with highlighted keywords
   */
  export const highlightText = (text, keywords) => {
    if (!text || !keywords || keywords.length === 0) {
      return text;
    }
  
    let highlightedText = text;
  
    // Sort keywords by length (longest first) to avoid issues
    // where shorter keywords are part of longer ones
    const sortedKeywords = [...keywords].sort((a, b) => b.length - a.length);
  
    sortedKeywords.forEach(keyword => {
      if (!keyword) return;
      
      const regex = new RegExp(`\\b${keyword}\\b`, 'gi');
      highlightedText = highlightedText.replace(
        regex,
        `<span class="text-red-600 font-bold">${keyword}</span>`
      );
    });
  
    return highlightedText;
  };
  
  /**
   * Extract keywords from a narrative response
   * 
   * @param {Object} narrativeResponse - Response from the narrative API
   * @returns {Array} - Array of keywords
   */
  export const extractKeywords = (narrativeResponse) => {
    if (!narrativeResponse || !narrativeResponse.keywords) {
      return [];
    }
  
    return narrativeResponse.keywords.map(k => 
      typeof k === 'object' && k.text ? k.text : k
    );
  };
  
  /**
   * Convert file size in bytes to human-readable format
   * 
   * @param {number} bytes - File size in bytes
   * @param {number} decimals - Decimal places for the result
   * @returns {string} - Human-readable file size
   */
  export const formatFileSize = (bytes, decimals = 1) => {
    if (bytes === 0) return '0 Bytes';
  
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
  
    const i = Math.floor(Math.log(bytes) / Math.log(k));
  
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
  };