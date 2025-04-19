import { useState, useCallback } from 'react';
import axios from 'axios';

/**
 * Custom hook to manage search functionality for memories
 * @param {string} [type='user'] - Type of memories to search ('user' or 'public')
 * @returns {Object} - An object containing search methods and state
 */
export const useSearch = (type = 'user') => {
  const [searchResults, setSearchResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  /**
   * Search memories by location
   * @param {string} location - Location to search for
   * @returns {Promise<Array>} - Array of matching memories
   */
  const searchByLocation = useCallback(async (location) => {
    if (!location) {
      setSearchResults([]);
      return [];
    }

    try {
      setLoading(true);
      setError(null);

      // Fetch memories by location
      const response = await axios.get(`/api/memories/search`, {
        params: { 
          location, 
          type 
        }
      });

      setSearchResults(response.data);
      return response.data;
    } catch (err) {
      setError(err.response?.data || 'An error occurred while searching memories');
      setSearchResults([]);
      return [];
    } finally {
      setLoading(false);
    }
  }, [type]);

  /**
   * Search memories by keywords
   * @param {string[]} keywords - Keywords to search for
   * @returns {Promise<Array>} - Array of matching memories
   */
  const searchByKeywords = useCallback(async (keywords) => {
    if (!keywords || keywords.length === 0) {
      setSearchResults([]);
      return [];
    }

    try {
      setLoading(true);
      setError(null);

      // Fetch memories by keywords
      const response = await axios.get(`/api/memories/search`, {
        params: { 
          keywords: keywords.join(','), 
          type 
        }
      });

      setSearchResults(response.data);
      return response.data;
    } catch (err) {
      setError(err.response?.data || 'An error occurred while searching memories');
      setSearchResults([]);
      return [];
    } finally {
      setLoading(false);
    }
  }, [type]);

  // Clear search results
  const clearSearch = () => {
    setSearchResults([]);
    setError(null);
  };

  return {
    searchResults,
    loading,
    error,
    searchByLocation,
    searchByKeywords,
    clearSearch
  };
};