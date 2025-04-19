import { useState, useCallback } from 'react';
import axios from 'axios';

/**
 * Custom hook to generate narratives about memories
 * @param {string} [type='user'] - Type of narrative to generate ('user' or 'public')
 * @returns {Object} - An object containing narrative generation methods and state
 */
export const useNarrative = (type = 'user') => {
  const [narrative, setNarrative] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  /**
   * Generate a narrative for a specific location
   * @param {string} location - Location to generate narrative for
   * @returns {Promise<Object>} - Generated narrative with text and keywords
   */
  const generateNarrative = useCallback(async (location) => {
    if (!location) {
      setError('Location is required');
      return null;
    }

    try {
      setLoading(true);
      setError(null);

      // Generate narrative via API
      const response = await axios.get(`/api/generate/${type}/${encodeURIComponent(location)}`);

      setNarrative(response.data);
      return response.data;
    } catch (err) {
      const errorMessage = err.response?.data?.detail || 'An error occurred while generating narrative';
      setError(errorMessage);
      setNarrative(null);
      return null;
    } finally {
      setLoading(false);
    }
  }, [type]);

  /**
   * Clear the current narrative
   */
  const clearNarrative = () => {
    setNarrative(null);
    setError(null);
  };

  return {
    narrative,
    loading,
    error,
    generateNarrative,
    clearNarrative
  };
};