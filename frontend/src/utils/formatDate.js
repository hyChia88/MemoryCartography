/**
 * Format a date string into a more readable format
 * @param {string} dateString - Date string in ISO format (YYYY-MM-DD)
 * @returns {string} - Formatted date string
 */
export const formatDate = (dateString) => {
    if (!dateString) return '';
    
    try {
      const date = new Date(dateString);
      return new Intl.DateTimeFormat('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric'
      }).format(date);
    } catch (error) {
      console.error('Error formatting date:', error);
      return dateString; // Return original string if formatting fails
    }
  };
  
  /**
   * Check if a date is within the last week
   * @param {string} dateString - Date string in ISO format
   * @returns {boolean} - True if date is within last week
   */
  export const isRecentDate = (dateString) => {
    if (!dateString) return false;
    
    try {
      const date = new Date(dateString);
      const now = new Date();
      const oneWeekAgo = new Date();
      oneWeekAgo.setDate(now.getDate() - 7);
      
      return date >= oneWeekAgo;
    } catch (error) {
      return false;
    }
  };
  
  /**
   * Format a date into a relative time string (today, yesterday, 3 days ago, etc.)
   * @param {string} dateString - Date string in ISO format
   * @returns {string} - Relative time string
   */
  export const getRelativeTimeString = (dateString) => {
    if (!dateString) return '';
    
    try {
      const date = new Date(dateString);
      const now = new Date();
      const yesterday = new Date();
      yesterday.setDate(now.getDate() - 1);
      
      // Check if it's today
      if (
        date.getDate() === now.getDate() &&
        date.getMonth() === now.getMonth() &&
        date.getFullYear() === now.getFullYear()
      ) {
        return 'Today';
      }
      
      // Check if it's yesterday
      if (
        date.getDate() === yesterday.getDate() &&
        date.getMonth() === yesterday.getMonth() &&
        date.getFullYear() === yesterday.getFullYear()
      ) {
        return 'Yesterday';
      }
      
      // Calculate days ago
      const diffTime = Math.abs(now - date);
      const diffDays = Math.floor(diffTime / (1000 * 60 * 60 * 24));
      
      if (diffDays < 7) {
        return `${diffDays} days ago`;
      }
      
      // Default formatting for older dates
      return formatDate(dateString);
    } catch (error) {
      console.error('Error calculating relative time:', error);
      return formatDate(dateString);
    }
  };